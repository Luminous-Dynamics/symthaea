//! Method-neutral Monte Carlo evaluator for clustered-IV inference benchmarks.
//!
//! The scorecard deliberately reports a vector of diagnostics. It does not
//! create a composite rank: a narrow confidence set cannot compensate for
//! invalid size/coverage, and a whole-line set cannot be presented as useful
//! localization merely because it covers every parameter.
//!
//! One input row represents one scenario × method/test × replicate result.
//! Test outcomes, point estimates, confidence sets and run failures are
//! separate fields. Confidence sets are unions of canonical closed intervals;
//! a missing lower endpoint means -infinity and a missing upper endpoint means
//! +infinity. An empty vector is the empty set.
//!
//! This crate is an in-memory Rust API. Adapters are responsible for parsing
//! their R/Stata/tool output and constructing these typed records. No external
//! inference package or third-party dependency is required.

use core::fmt;
use std::collections::{BTreeMap, HashSet};

pub const SCORECARD_SCHEMA_VERSION: &str = "1.0.0";
pub const WILSON_95_Z: f64 = 1.959_963_984_540_054;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HypothesisRegime {
    TrueNull,
    Alternative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    Completed,
    Failed,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConfidenceSetTopology {
    Empty,
    Bounded,
    LeftRay,
    RightRay,
    TwoRays,
    Disconnected,
    WholeLine,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceInterval {
    /// None represents negative infinity.
    pub lower: Option<f64>,
    /// None represents positive infinity.
    pub upper: Option<f64>,
}

impl ConfidenceInterval {
    pub fn contains(&self, value: f64) -> bool {
        self.lower.map_or(true, |lower| lower <= value)
            && self.upper.map_or(true, |upper| value <= upper)
    }

    pub fn finite_width(&self) -> Option<f64> {
        match (self.lower, self.upper) {
            (Some(lower), Some(upper)) => Some(upper - lower),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConfidenceSet {
    /// Ordered, non-overlapping union of closed intervals.
    pub intervals: Vec<ConfidenceInterval>,
}

impl ConfidenceSet {
    pub fn empty() -> Self {
        Self { intervals: Vec::new() }
    }

    pub fn whole_line() -> Self {
        Self {
            intervals: vec![ConfidenceInterval {
                lower: None,
                upper: None,
            }],
        }
    }

    /// Validate a canonical interval union. Adjacent or overlapping pieces
    /// must be merged by the adapter before scoring.
    pub fn validate(&self) -> Result<(), ScorecardError> {
        for (index, interval) in self.intervals.iter().enumerate() {
            if let Some(lower) = interval.lower {
                if !lower.is_finite() {
                    return Err(ScorecardError::NonFiniteIntervalEndpoint { index });
                }
            }
            if let Some(upper) = interval.upper {
                if !upper.is_finite() {
                    return Err(ScorecardError::NonFiniteIntervalEndpoint { index });
                }
            }
            if let (Some(lower), Some(upper)) = (interval.lower, interval.upper) {
                if lower > upper {
                    return Err(ScorecardError::ReversedInterval { index });
                }
                if !(upper - lower).is_finite() {
                    return Err(ScorecardError::NumericalOverflow {
                        field: "confidence-set interval width",
                    });
                }
            }
        }

        for (index, pair) in self.intervals.windows(2).enumerate() {
            let previous = pair[0];
            let next = pair[1];
            let previous_upper = previous
                .upper
                .ok_or(ScorecardError::NonCanonicalIntervalUnion { index })?;
            let next_lower = next
                .lower
                .ok_or(ScorecardError::NonCanonicalIntervalUnion { index })?;
            if next_lower <= previous_upper {
                return Err(ScorecardError::OverlappingOrAdjacentIntervals { index });
            }
        }

        if let Some(width) = self.finite_width() {
            if !width.is_finite() {
                return Err(ScorecardError::NumericalOverflow {
                    field: "total confidence-set width",
                });
            }
        }
        Ok(())
    }

    pub fn contains(&self, value: f64) -> bool {
        self.intervals.iter().any(|interval| interval.contains(value))
    }

    pub fn topology(&self) -> ConfidenceSetTopology {
        match self.intervals.as_slice() {
            [] => ConfidenceSetTopology::Empty,
            [only] => match (only.lower, only.upper) {
                (None, None) => ConfidenceSetTopology::WholeLine,
                (None, Some(_)) => ConfidenceSetTopology::LeftRay,
                (Some(_), None) => ConfidenceSetTopology::RightRay,
                (Some(_), Some(_)) => ConfidenceSetTopology::Bounded,
            },
            [first, second]
                if first.lower.is_none() && second.upper.is_none() =>
            {
                ConfidenceSetTopology::TwoRays
            }
            _ => ConfidenceSetTopology::Disconnected,
        }
    }

    /// Returns None for empty or unbounded sets. For a bounded union, width is
    /// the sum of lengths of all components, including disconnected components.
    pub fn finite_width(&self) -> Option<f64> {
        if self.intervals.is_empty()
            || self
                .intervals
                .iter()
                .any(|interval| interval.lower.is_none() || interval.upper.is_none())
        {
            return None;
        }
        Some(
            self.intervals
                .iter()
                .map(|interval| interval.finite_width().unwrap_or(0.0))
                .sum(),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReplicationResult {
    pub scenario_id: String,
    /// Include the estimator and inferential variant, e.g.
    /// clusteriv020_cjar_crossfit_chisq or stata19_clr_cluster.
    pub method_id: String,
    pub replication_id: String,
    pub seed: u64,
    pub regime: HypothesisRegime,
    /// Structural truth under the DGP.
    pub true_beta: f64,
    /// Value in the tested structural null H0: beta = tested_null_beta.
    pub tested_null_beta: f64,
    pub status: RunStatus,
    /// P-value for the pre-registered structural null in this DGP replicate.
    pub p_value: Option<f64>,
    pub point_estimate: Option<f64>,
    /// The complete set, not a finite-grid approximation to its topology.
    pub confidence_set: Option<ConfidenceSet>,
    pub runtime_micros: Option<u64>,
}

impl ReplicationResult {
    pub fn validate(&self) -> Result<(), ScorecardError> {
        for (field, value) in [
            ("scenario_id", self.scenario_id.as_str()),
            ("method_id", self.method_id.as_str()),
            ("replication_id", self.replication_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ScorecardError::EmptyIdentifier { field });
            }
        }
        if !self.true_beta.is_finite() {
            return Err(ScorecardError::NonFiniteParameter { field: "true_beta" });
        }
        if !self.tested_null_beta.is_finite() {
            return Err(ScorecardError::NonFiniteParameter { field: "tested_null_beta" });
        }
        match self.regime {
            HypothesisRegime::TrueNull if self.true_beta != self.tested_null_beta => {
                return Err(ScorecardError::RegimeTruthMismatch {
                    regime: self.regime,
                    true_beta: self.true_beta,
                    tested_null_beta: self.tested_null_beta,
                });
            }
            HypothesisRegime::Alternative if self.true_beta == self.tested_null_beta => {
                return Err(ScorecardError::RegimeTruthMismatch {
                    regime: self.regime,
                    true_beta: self.true_beta,
                    tested_null_beta: self.tested_null_beta,
                });
            }
            _ => {}
        }
        if let Some(p_value) = self.p_value {
            if !p_value.is_finite() || !(0.0..=1.0).contains(&p_value) {
                return Err(ScorecardError::InvalidPValue { value: p_value });
            }
        }
        if let Some(estimate) = self.point_estimate {
            if !estimate.is_finite() {
                return Err(ScorecardError::NonFiniteParameter {
                    field: "point_estimate",
                });
            }
        }
        if let Some(confidence_set) = &self.confidence_set {
            confidence_set.validate()?;
        }

        match self.status {
            RunStatus::Completed => {
                if self.p_value.is_none()
                    && self.point_estimate.is_none()
                    && self.confidence_set.is_none()
                {
                    return Err(ScorecardError::CompletedWithoutMetrics);
                }
            }
            RunStatus::Failed | RunStatus::NotApplicable => {
                if self.p_value.is_some()
                    || self.point_estimate.is_some()
                    || self.confidence_set.is_some()
                {
                    return Err(ScorecardError::NonCompletedRunHasMetrics);
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WilsonInterval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConfidenceSetTopologyCounts {
    pub empty: usize,
    pub bounded: usize,
    pub left_ray: usize,
    pub right_ray: usize,
    pub two_rays: usize,
    pub disconnected: usize,
    pub whole_line: usize,
}

impl ConfidenceSetTopologyCounts {
    fn add(&mut self, topology: ConfidenceSetTopology) {
        match topology {
            ConfidenceSetTopology::Empty => self.empty += 1,
            ConfidenceSetTopology::Bounded => self.bounded += 1,
            ConfidenceSetTopology::LeftRay => self.left_ray += 1,
            ConfidenceSetTopology::RightRay => self.right_ray += 1,
            ConfidenceSetTopology::TwoRays => self.two_rays += 1,
            ConfidenceSetTopology::Disconnected => self.disconnected += 1,
            ConfidenceSetTopology::WholeLine => self.whole_line += 1,
        }
    }

    pub fn total(&self) -> usize {
        self.empty
            + self.bounded
            + self.left_ray
            + self.right_ray
            + self.two_rays
            + self.disconnected
            + self.whole_line
    }

}

#[derive(Debug, Clone, PartialEq)]
pub struct GroupSummary {
    pub scenario_id: String,
    pub method_id: String,
    pub regime: HypothesisRegime,
    pub true_beta: f64,
    pub tested_null_beta: f64,

    /// Attempted excludes explicitly NOT_APPLICABLE rows.
    pub attempted_replicates: usize,
    pub completed_replicates: usize,
    pub failed_replicates: usize,
    pub not_applicable_replicates: usize,
    pub failure_rate: Option<f64>,

    pub p_value_replicates: usize,
    pub rejected_replicates: usize,
    pub rejection_rate: Option<f64>,
    pub rejection_wilson_95: Option<WilsonInterval>,
    /// Populated only for true-null cells; None for alternatives.
    pub empirical_size: Option<f64>,
    pub size_wilson_95: Option<WilsonInterval>,
    /// Populated only for alternative cells; None for true-null cells.
    pub empirical_power: Option<f64>,
    pub power_wilson_95: Option<WilsonInterval>,
    /// Missing p-values are all treated as non-rejections for the lower bound
    /// and as rejections for the upper bound.
    pub rejection_rate_missing_all_nonreject: Option<f64>,
    pub rejection_rate_missing_all_reject: Option<f64>,
    pub missing_p_value_replicates: usize,
    /// Ten equal-width bins over [0,1], with p=1 in the last bin.
    pub p_value_histogram_10: [usize; 10],

    pub estimate_replicates: usize,
    pub mean_estimate: Option<f64>,
    pub bias: Option<f64>,
    pub rmse: Option<f64>,

    pub confidence_set_replicates: usize,
    pub covered_replicates: usize,
    pub coverage_rate: Option<f64>,
    pub coverage_wilson_95: Option<WilsonInterval>,
    /// Missing confidence sets are treated as uncovered / covered respectively.
    pub coverage_rate_missing_uncovered: Option<f64>,
    pub coverage_rate_missing_covered: Option<f64>,
    pub missing_confidence_set_replicates: usize,
    pub topology_counts: ConfidenceSetTopologyCounts,
    /// Counts any set with at least one infinite endpoint, independently of
    /// topology (a disconnected union can be bounded or partly unbounded).
    pub unbounded_confidence_set_replicates: usize,
    pub finite_width_replicates: usize,
    pub median_finite_set_width: Option<f64>,
    pub p90_finite_set_width: Option<f64>,

    pub runtime_replicates: usize,
    pub median_runtime_micros: Option<f64>,
    pub p95_runtime_micros: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScorecardError {
    EmptyIdentifier { field: &'static str },
    NonFiniteParameter { field: &'static str },
    InvalidPValue { value: f64 },
    InvalidAlpha { value: f64 },
    InvalidCriticalValue { value: f64 },
    NonFiniteIntervalEndpoint { index: usize },
    ReversedInterval { index: usize },
    OverlappingOrAdjacentIntervals { index: usize },
    NonCanonicalIntervalUnion { index: usize },
    CompletedWithoutMetrics,
    NonCompletedRunHasMetrics,
    DuplicateReplicationId { scenario_id: String, method_id: String, replication_id: String },
    InconsistentTrueBeta { scenario_id: String, method_id: String },
    InconsistentTestedNullBeta { scenario_id: String, method_id: String },
    RegimeTruthMismatch { regime: HypothesisRegime, true_beta: f64, tested_null_beta: f64 },
    NumericalOverflow { field: &'static str },
}

impl fmt::Display for ScorecardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier { field } => write!(f, "{field} must not be empty"),
            Self::NonFiniteParameter { field } => write!(f, "{field} must be finite"),
            Self::InvalidPValue { value } => write!(f, "p-value must lie in [0,1]: {value}"),
            Self::InvalidAlpha { value } => write!(f, "alpha must lie strictly inside (0,1): {value}"),
            Self::InvalidCriticalValue { value } => write!(f, "Wilson z critical value must be finite and positive: {value}"),
            Self::NonFiniteIntervalEndpoint { index } => write!(f, "interval {index} has a non-finite finite endpoint"),
            Self::ReversedInterval { index } => write!(f, "interval {index} has lower endpoint above upper endpoint"),
            Self::OverlappingOrAdjacentIntervals { index } => write!(f, "confidence-set intervals {index} and {} overlap or are adjacent; merge them", index + 1),
            Self::NonCanonicalIntervalUnion { index } => write!(f, "confidence-set intervals near {index} are not sorted/canonical"),
            Self::CompletedWithoutMetrics => write!(f, "completed run has no p-value, point estimate or confidence set"),
            Self::NonCompletedRunHasMetrics => write!(f, "failed/not-applicable run must not carry inferential metrics"),
            Self::DuplicateReplicationId { scenario_id, method_id, replication_id } => write!(f, "duplicate replicate {replication_id:?} for scenario {scenario_id:?} and method {method_id:?}"),
            Self::InconsistentTrueBeta { scenario_id, method_id } => write!(f, "true beta differs within scenario {scenario_id:?} / method {method_id:?} group"),
            Self::InconsistentTestedNullBeta { scenario_id, method_id } => write!(f, "tested null beta differs within scenario {scenario_id:?} / method {method_id:?} group"),
            Self::RegimeTruthMismatch { regime, true_beta, tested_null_beta } => write!(f, "regime {regime:?} does not match true beta {true_beta} and tested null beta {tested_null_beta}"),
            Self::NumericalOverflow { field } => write!(f, "numerical overflow computing {field}"),
        }
    }
}
impl std::error::Error for ScorecardError {}

/// Wilson score interval for a binomial proportion.
///
/// The correction is for Monte Carlo uncertainty in the estimated rate; it is
/// not a correction to the inferential p-values being evaluated.
pub fn wilson_interval(
    successes: usize,
    trials: usize,
    z_critical: f64,
) -> Result<Option<WilsonInterval>, ScorecardError> {
    if !z_critical.is_finite() || z_critical <= 0.0 {
        return Err(ScorecardError::InvalidCriticalValue { value: z_critical });
    }
    if trials == 0 {
        return Ok(None);
    }
    if successes > trials {
        return Err(ScorecardError::NumericalOverflow {
            field: "binomial successes/trials",
        });
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = z_critical * z_critical;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = z_critical
        * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()
        / denominator;
    let lower = (center - half_width).max(0.0);
    let upper = (center + half_width).min(1.0);
    if !lower.is_finite() || !upper.is_finite() {
        return Err(ScorecardError::NumericalOverflow {
            field: "Wilson interval",
        });
    }
    Ok(Some(WilsonInterval { lower, upper }))
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GroupKey {
    scenario_id: String,
    method_id: String,
    regime: HypothesisRegime,
}

fn quantile(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let position = (sorted.len() - 1) as f64 * probability;
    let lower_index = position.floor() as usize;
    let upper_index = position.ceil() as usize;
    let fraction = position - lower_index as f64;
    Some(
        sorted[lower_index] * (1.0 - fraction)
            + sorted[upper_index] * fraction,
    )
}

/// Aggregate normalized replicate outputs. Grouping is deterministic and
/// independent of input row order. Null and alternative regimes are never
/// pooled. NOT_APPLICABLE rows are counted but excluded from attempted runs.
pub fn aggregate_scorecard(
    records: &[ReplicationResult],
    alpha: f64,
) -> Result<Vec<GroupSummary>, ScorecardError> {
    aggregate_scorecard_with_critical(records, alpha, WILSON_95_Z)
}

pub fn aggregate_scorecard_with_critical(
    records: &[ReplicationResult],
    alpha: f64,
    z_critical: f64,
) -> Result<Vec<GroupSummary>, ScorecardError> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(ScorecardError::InvalidAlpha { value: alpha });
    }
    if !z_critical.is_finite() || z_critical <= 0.0 {
        return Err(ScorecardError::InvalidCriticalValue { value: z_critical });
    }

    let mut groups: BTreeMap<GroupKey, Vec<&ReplicationResult>> = BTreeMap::new();
    for record in records {
        record.validate()?;
        let key = GroupKey {
            scenario_id: record.scenario_id.clone(),
            method_id: record.method_id.clone(),
            regime: record.regime,
        };
        groups.entry(key).or_default().push(record);
    }

    let mut summaries = Vec::with_capacity(groups.len());
    for (key, mut group) in groups {
        group.sort_by(|left, right| {
            left.replication_id
                .cmp(&right.replication_id)
                .then_with(|| left.seed.cmp(&right.seed))
        });

        let mut seen = HashSet::with_capacity(group.len());
        let true_beta = group[0].true_beta;
        let tested_null_beta = group[0].tested_null_beta;
        let mut completed = 0usize;
        let mut failed = 0usize;
        let mut not_applicable = 0usize;
        let mut p_value_count = 0usize;
        let mut rejected = 0usize;
        let mut histogram = [0usize; 10];
        let mut estimate_errors = Vec::new();
        let mut estimate_sum = 0.0;
        let mut squared_error_sum = 0.0;
        let mut confidence_count = 0usize;
        let mut covered = 0usize;
        let mut topology_counts = ConfidenceSetTopologyCounts::default();
        let mut unbounded_set_count = 0usize;
        let mut finite_widths = Vec::new();
        let mut runtimes = Vec::new();

        for record in &group {
            if !seen.insert(record.replication_id.as_str()) {
                return Err(ScorecardError::DuplicateReplicationId {
                    scenario_id: key.scenario_id.clone(),
                    method_id: key.method_id.clone(),
                    replication_id: record.replication_id.clone(),
                });
            }
            if record.true_beta != true_beta {
                return Err(ScorecardError::InconsistentTrueBeta {
                    scenario_id: key.scenario_id.clone(),
                    method_id: key.method_id.clone(),
                });
            }
            if record.tested_null_beta != tested_null_beta {
                return Err(ScorecardError::InconsistentTestedNullBeta {
                    scenario_id: key.scenario_id.clone(),
                    method_id: key.method_id.clone(),
                });
            }

            match record.status {
                RunStatus::Completed => completed += 1,
                RunStatus::Failed => failed += 1,
                RunStatus::NotApplicable => not_applicable += 1,
            }
            if let Some(runtime) = record.runtime_micros {
                runtimes.push(runtime as f64);
            }
            if record.status != RunStatus::Completed {
                continue;
            }

            if let Some(p_value) = record.p_value {
                p_value_count += 1;
                if p_value <= alpha {
                    rejected += 1;
                }
                let bin = if p_value == 1.0 {
                    9
                } else {
                    ((p_value * 10.0).floor() as usize).min(9)
                };
                histogram[bin] += 1;
            }

            if let Some(estimate) = record.point_estimate {
                let error = estimate - true_beta;
                if !error.is_finite() {
                    return Err(ScorecardError::NumericalOverflow {
                        field: "point-estimate error",
                    });
                }
                estimate_errors.push(error);
                estimate_sum += estimate;
                squared_error_sum += error * error;
            }

            if let Some(confidence_set) = &record.confidence_set {
                confidence_count += 1;
                let topology = confidence_set.topology();
                topology_counts.add(topology);
                if confidence_set.intervals.iter().any(|interval| {
                    interval.lower.is_none() || interval.upper.is_none()
                }) {
                    unbounded_set_count += 1;
                }
                if confidence_set.contains(true_beta) {
                    covered += 1;
                }
                if let Some(width) = confidence_set.finite_width() {
                    finite_widths.push(width);
                }
            }
        }

        let attempted = completed + failed;
        let missing_p = attempted.saturating_sub(p_value_count);
        let missing_set = attempted.saturating_sub(confidence_count);

        let rejection_rate = if p_value_count > 0 {
            Some(rejected as f64 / p_value_count as f64)
        } else {
            None
        };
        let rejection_wilson_95 =
            wilson_interval(rejected, p_value_count, z_critical)?;
        let empirical_size = if key.regime == HypothesisRegime::TrueNull {
            rejection_rate
        } else {
            None
        };
        let empirical_power = if key.regime == HypothesisRegime::Alternative {
            rejection_rate
        } else {
            None
        };
        let size_wilson_95 = if key.regime == HypothesisRegime::TrueNull {
            rejection_wilson_95
        } else {
            None
        };
        let power_wilson_95 = if key.regime == HypothesisRegime::Alternative {
            rejection_wilson_95
        } else {
            None
        };
        let rejection_rate_missing_all_nonreject = if attempted > 0 {
            Some(rejected as f64 / attempted as f64)
        } else {
            None
        };
        let rejection_rate_missing_all_reject = if attempted > 0 {
            Some((rejected + missing_p) as f64 / attempted as f64)
        } else {
            None
        };

        let coverage_rate = if confidence_count > 0 {
            Some(covered as f64 / confidence_count as f64)
        } else {
            None
        };
        let coverage_wilson_95 =
            wilson_interval(covered, confidence_count, z_critical)?;
        // Do not manufacture coverage bounds for estimators that did not
        // emit any confidence sets at all. When at least one set exists,
        // missing/failed outputs are conservatively bounded over all attempts.
        let (coverage_rate_missing_uncovered, coverage_rate_missing_covered) =
            if attempted > 0 && confidence_count > 0 {
                (
                    Some(covered as f64 / attempted as f64),
                    Some((covered + missing_set) as f64 / attempted as f64),
                )
            } else {
                (None, None)
            };

        if !estimate_sum.is_finite() || !squared_error_sum.is_finite() {
            return Err(ScorecardError::NumericalOverflow {
                field: "point-estimate aggregate",
            });
        }
        let estimate_count = estimate_errors.len();
        let mean_estimate = if estimate_count > 0 {
            Some(estimate_sum / estimate_count as f64)
        } else {
            None
        };
        let bias = mean_estimate.map(|mean| mean - true_beta);
        let rmse = if estimate_count > 0 {
            Some((squared_error_sum / estimate_count as f64).sqrt())
        } else {
            None
        };
        if mean_estimate.is_some_and(|value| !value.is_finite())
            || bias.is_some_and(|value| !value.is_finite())
            || rmse.is_some_and(|value| !value.is_finite())
        {
            return Err(ScorecardError::NumericalOverflow {
                field: "point-estimate summary",
            });
        }

        finite_widths.sort_by(f64::total_cmp);
        runtimes.sort_by(f64::total_cmp);
        let finite_width_count = finite_widths.len();
        let runtime_count = runtimes.len();
        let failure_rate = if attempted > 0 {
            Some(failed as f64 / attempted as f64)
        } else {
            None
        };

        summaries.push(GroupSummary {
            scenario_id: key.scenario_id,
            method_id: key.method_id,
            regime: key.regime,
            true_beta,
            tested_null_beta,
            attempted_replicates: attempted,
            completed_replicates: completed,
            failed_replicates: failed,
            not_applicable_replicates: not_applicable,
            failure_rate,
            p_value_replicates: p_value_count,
            rejected_replicates: rejected,
            rejection_rate,
            rejection_wilson_95,
            empirical_size,
            size_wilson_95,
            empirical_power,
            power_wilson_95,
            rejection_rate_missing_all_nonreject,
            rejection_rate_missing_all_reject,
            missing_p_value_replicates: missing_p,
            p_value_histogram_10: histogram,
            estimate_replicates: estimate_count,
            mean_estimate,
            bias,
            rmse,
            confidence_set_replicates: confidence_count,
            covered_replicates: covered,
            coverage_rate,
            coverage_wilson_95,
            coverage_rate_missing_uncovered,
            coverage_rate_missing_covered,
            missing_confidence_set_replicates: missing_set,
            topology_counts,
            unbounded_confidence_set_replicates: unbounded_set_count,
            finite_width_replicates: finite_width_count,
            median_finite_set_width: quantile(&finite_widths, 0.50),
            p90_finite_set_width: quantile(&finite_widths, 0.90),
            runtime_replicates: runtime_count,
            median_runtime_micros: quantile(&runtimes, 0.50),
            p95_runtime_micros: quantile(&runtimes, 0.95),
        });
    }

    Ok(summaries)
}


fn json_escape(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for ch in value.chars() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            c if c.is_control() => output.push_str(&format!("\\u{:04x}", c as u32)),
            c => output.push(c),
        }
    }
    output.push('"');
    output
}

fn json_f64(value: f64) -> String {
    format!("{:.17e}", value)
}

fn json_option_f64(value: Option<f64>) -> String {
    value.map(json_f64).unwrap_or_else(|| "null".to_owned())
}

fn json_wilson(value: Option<WilsonInterval>) -> String {
    match value {
        Some(interval) => format!(
            "{{\"lower\":{},\"upper\":{}}}",
            json_f64(interval.lower),
            json_f64(interval.upper),
        ),
        None => "null".to_owned(),
    }
}

impl GroupSummary {
    pub fn to_json_object(&self) -> String {
        let mut fields = Vec::with_capacity(64);
        fields.push(format!("\"scenario_id\":{}", json_escape(&self.scenario_id)));
        fields.push(format!("\"method_id\":{}", json_escape(&self.method_id)));
        fields.push(format!(
            "\"regime\":{}",
            json_escape(match self.regime {
                HypothesisRegime::TrueNull => "true_null",
                HypothesisRegime::Alternative => "alternative",
            })
        ));
        fields.push(format!("\"true_beta\":{}", json_f64(self.true_beta)));
        fields.push(format!("\"tested_null_beta\":{}", json_f64(self.tested_null_beta)));
        fields.push(format!("\"attempted_replicates\":{}", self.attempted_replicates));
        fields.push(format!("\"completed_replicates\":{}", self.completed_replicates));
        fields.push(format!("\"failed_replicates\":{}", self.failed_replicates));
        fields.push(format!("\"not_applicable_replicates\":{}", self.not_applicable_replicates));
        fields.push(format!("\"failure_rate\":{}", json_option_f64(self.failure_rate)));
        fields.push(format!("\"p_value_replicates\":{}", self.p_value_replicates));
        fields.push(format!("\"rejected_replicates\":{}", self.rejected_replicates));
        fields.push(format!("\"rejection_rate\":{}", json_option_f64(self.rejection_rate)));
        fields.push(format!("\"rejection_wilson_95\":{}", json_wilson(self.rejection_wilson_95)));
        fields.push(format!("\"empirical_size\":{}", json_option_f64(self.empirical_size)));
        fields.push(format!("\"size_wilson_95\":{}", json_wilson(self.size_wilson_95)));
        fields.push(format!("\"empirical_power\":{}", json_option_f64(self.empirical_power)));
        fields.push(format!("\"power_wilson_95\":{}", json_wilson(self.power_wilson_95)));
        fields.push(format!(
            "\"rejection_rate_missing_all_nonreject\":{}",
            json_option_f64(self.rejection_rate_missing_all_nonreject)
        ));
        fields.push(format!(
            "\"rejection_rate_missing_all_reject\":{}",
            json_option_f64(self.rejection_rate_missing_all_reject)
        ));
        fields.push(format!("\"missing_p_value_replicates\":{}", self.missing_p_value_replicates));
        fields.push(format!(
            "\"p_value_histogram_10\":[{}]",
            self.p_value_histogram_10.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
        ));
        fields.push(format!("\"estimate_replicates\":{}", self.estimate_replicates));
        fields.push(format!("\"mean_estimate\":{}", json_option_f64(self.mean_estimate)));
        fields.push(format!("\"bias\":{}", json_option_f64(self.bias)));
        fields.push(format!("\"rmse\":{}", json_option_f64(self.rmse)));
        fields.push(format!("\"confidence_set_replicates\":{}", self.confidence_set_replicates));
        fields.push(format!("\"covered_replicates\":{}", self.covered_replicates));
        fields.push(format!("\"coverage_rate\":{}", json_option_f64(self.coverage_rate)));
        fields.push(format!("\"coverage_wilson_95\":{}", json_wilson(self.coverage_wilson_95)));
        fields.push(format!(
            "\"coverage_rate_missing_uncovered\":{}",
            json_option_f64(self.coverage_rate_missing_uncovered)
        ));
        fields.push(format!(
            "\"coverage_rate_missing_covered\":{}",
            json_option_f64(self.coverage_rate_missing_covered)
        ));
        fields.push(format!("\"missing_confidence_set_replicates\":{}", self.missing_confidence_set_replicates));
        fields.push(format!(
            "\"topology_counts\":{{\"empty\":{},\"bounded\":{},\"left_ray\":{},\"right_ray\":{},\"two_rays\":{},\"disconnected\":{},\"whole_line\":{}}}",
            self.topology_counts.empty,
            self.topology_counts.bounded,
            self.topology_counts.left_ray,
            self.topology_counts.right_ray,
            self.topology_counts.two_rays,
            self.topology_counts.disconnected,
            self.topology_counts.whole_line,
        ));
        fields.push(format!(
            "\"unbounded_confidence_set_replicates\":{}",
            self.unbounded_confidence_set_replicates
        ));
        fields.push(format!("\"finite_width_replicates\":{}", self.finite_width_replicates));
        fields.push(format!(
            "\"median_finite_set_width\":{}",
            json_option_f64(self.median_finite_set_width)
        ));
        fields.push(format!(
            "\"p90_finite_set_width\":{}",
            json_option_f64(self.p90_finite_set_width)
        ));
        fields.push(format!("\"runtime_replicates\":{}", self.runtime_replicates));
        fields.push(format!(
            "\"median_runtime_micros\":{}",
            json_option_f64(self.median_runtime_micros)
        ));
        fields.push(format!(
            "\"p95_runtime_micros\":{}",
            json_option_f64(self.p95_runtime_micros)
        ));
        format!("{{{}}}", fields.join(","))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScorecardReport {
    pub schema_version: &'static str,
    pub alpha: f64,
    pub wilson_z_critical: f64,
    pub groups: Vec<GroupSummary>,
}

impl ScorecardReport {
    /// Stable JSON export; absent values are serialized as JSON null.
    pub fn to_json(&self) -> String {
        let groups = self.groups.iter()
            .map(GroupSummary::to_json_object)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "{{\"schema_version\":{},\"alpha\":{},",
                "\"wilson_z_critical\":{},\"groups\":[{}]}}\n"
            ),
            json_escape(self.schema_version),
            json_f64(self.alpha),
            json_f64(self.wilson_z_critical),
            groups,
        )
    }
}

/// Aggregate and package the scorecard into a stable report structure.
pub fn aggregate_scorecard_report(
    records: &[ReplicationResult],
    alpha: f64,
) -> Result<ScorecardReport, ScorecardError> {
    aggregate_scorecard_report_with_critical(records, alpha, WILSON_95_Z)
}

pub fn aggregate_scorecard_report_with_critical(
    records: &[ReplicationResult],
    alpha: f64,
    z_critical: f64,
) -> Result<ScorecardReport, ScorecardError> {
    let groups = aggregate_scorecard_with_critical(records, alpha, z_critical)?;
    Ok(ScorecardReport {
        schema_version: SCORECARD_SCHEMA_VERSION,
        alpha,
        wilson_z_critical: z_critical,
        groups,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(lower: Option<f64>, upper: Option<f64>) -> ConfidenceInterval {
        ConfidenceInterval { lower, upper }
    }

    fn record(
        id: &str,
        regime: HypothesisRegime,
        status: RunStatus,
        p_value: Option<f64>,
        estimate: Option<f64>,
        confidence_set: Option<ConfidenceSet>,
    ) -> ReplicationResult {
        ReplicationResult {
            scenario_id: "scenario-a".to_owned(),
            method_id: "method-a".to_owned(),
            replication_id: id.to_owned(),
            seed: id.parse::<u64>().unwrap_or(1),
            regime,
            true_beta: 2.0,
            tested_null_beta: if regime == HypothesisRegime::TrueNull { 2.0 } else { 0.0 },
            status,
            p_value,
            point_estimate: estimate,
            confidence_set,
            runtime_micros: Some(100),
        }
    }

    fn bounded(lower: f64, upper: f64) -> ConfidenceSet {
        ConfidenceSet {
            intervals: vec![interval(Some(lower), Some(upper))],
        }
    }

    #[test]
    fn wilson_interval_known_answer_for_five_of_one_hundred() {
        let result = wilson_interval(5, 100, WILSON_95_Z).unwrap().unwrap();
        assert!((result.lower - 0.02154367915436796).abs() < 1.0e-14);
        assert!((result.upper - 0.11175046923191913).abs() < 1.0e-14);
    }

    #[test]
    fn wilson_zero_successes_still_has_positive_upper_bound() {
        let result = wilson_interval(0, 100, WILSON_95_Z).unwrap().unwrap();
        assert_eq!(result.lower, 0.0);
        assert!(result.upper > 0.03 && result.upper < 0.04);
    }

    #[test]
    fn wilson_all_successes_still_has_lower_uncertainty() {
        let result = wilson_interval(100, 100, WILSON_95_Z).unwrap().unwrap();
        assert!(result.lower > 0.96 && result.lower < 0.97);
        assert_eq!(result.upper, 1.0);
    }

    #[test]
    fn wilson_zero_trials_returns_none() {
        assert_eq!(wilson_interval(0, 0, WILSON_95_Z).unwrap(), None);
    }

    #[test]
    fn wilson_rejects_successes_above_trials() {
        assert!(matches!(
            wilson_interval(2, 1, WILSON_95_Z),
            Err(ScorecardError::NumericalOverflow { .. })
        ));
    }

    #[test]
    fn empty_set_is_empty_has_no_coverage_and_no_width() {
        let set = ConfidenceSet::empty();
        set.validate().unwrap();
        assert_eq!(set.topology(), ConfidenceSetTopology::Empty);
        assert!(!set.contains(2.0));
        assert_eq!(set.finite_width(), None);
    }

    #[test]
    fn bounded_set_covers_closed_endpoints_and_has_finite_width() {
        let set = bounded(1.0, 3.5);
        set.validate().unwrap();
        assert_eq!(set.topology(), ConfidenceSetTopology::Bounded);
        assert!(set.contains(1.0));
        assert!(set.contains(3.5));
        assert!(!set.contains(3.500_001));
        assert_eq!(set.finite_width(), Some(2.5));
    }

    #[test]
    fn left_and_right_rays_are_distinct_and_unbounded() {
        let left = ConfidenceSet { intervals: vec![interval(None, Some(2.0))] };
        let right = ConfidenceSet { intervals: vec![interval(Some(2.0), None)] };
        assert_eq!(left.topology(), ConfidenceSetTopology::LeftRay);
        assert_eq!(right.topology(), ConfidenceSetTopology::RightRay);
        assert_eq!(left.finite_width(), None);
        assert_eq!(right.finite_width(), None);
        assert!(left.contains(-1.0));
        assert!(!left.contains(3.0));
        assert!(right.contains(3.0));
        assert!(!right.contains(1.0));
    }

    #[test]
    fn two_rays_and_whole_line_keep_their_topology() {
        let two = ConfidenceSet {
            intervals: vec![interval(None, Some(1.0)), interval(Some(3.0), None)],
        };
        let whole = ConfidenceSet::whole_line();
        assert_eq!(two.topology(), ConfidenceSetTopology::TwoRays);
        assert_eq!(whole.topology(), ConfidenceSetTopology::WholeLine);
        assert!(two.contains(-10.0) && two.contains(10.0));
        assert!(!two.contains(2.0));
        assert!(whole.contains(-1.0) && whole.contains(1.0e9));
    }

    #[test]
    fn bounded_disconnected_union_uses_total_component_width() {
        let set = ConfidenceSet {
            intervals: vec![interval(Some(0.0), Some(1.0)), interval(Some(3.0), Some(5.0))],
        };
        set.validate().unwrap();
        assert_eq!(set.topology(), ConfidenceSetTopology::Disconnected);
        assert_eq!(set.finite_width(), Some(3.0));
        assert!(set.contains(0.5) && set.contains(4.0));
        assert!(!set.contains(2.0));
    }

    #[test]
    fn reversed_interval_fails_closed() {
        let set = bounded(3.0, 1.0);
        assert!(matches!(set.validate(), Err(ScorecardError::ReversedInterval { index: 0 })));
    }

    #[test]
    fn nan_endpoint_fails_closed() {
        let set = bounded(f64::NAN, 1.0);
        assert!(matches!(set.validate(), Err(ScorecardError::NonFiniteIntervalEndpoint { index: 0 })));
    }

    #[test]
    fn overlapping_or_adjacent_intervals_must_be_merged() {
        let overlap = ConfidenceSet {
            intervals: vec![interval(Some(0.0), Some(2.0)), interval(Some(1.0), Some(3.0))],
        };
        let adjacent = ConfidenceSet {
            intervals: vec![interval(Some(0.0), Some(2.0)), interval(Some(2.0), Some(3.0))],
        };
        assert!(matches!(overlap.validate(), Err(ScorecardError::OverlappingOrAdjacentIntervals { .. })));
        assert!(matches!(adjacent.validate(), Err(ScorecardError::OverlappingOrAdjacentIntervals { .. })));
    }

    #[test]
    fn unbounded_interval_must_be_first_or_last_in_canonical_order() {
        let set = ConfidenceSet {
            intervals: vec![interval(Some(0.0), Some(1.0)), interval(None, Some(4.0))],
        };
        assert!(matches!(set.validate(), Err(ScorecardError::NonCanonicalIntervalUnion { .. })));
    }

    #[test]
    fn null_dgp_rejection_rate_is_reported_as_size() {
        let records = vec![
            record("1", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.01), None, None),
            record("2", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.04), None, None),
            record("3", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.20), None, None),
            record("4", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.90), None, None),
        ];
        let output = aggregate_scorecard(&records, 0.05).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].attempted_replicates, 4);
        assert_eq!(output[0].rejected_replicates, 2);
        assert_eq!(output[0].rejection_rate, Some(0.5));
        assert_eq!(output[0].empirical_size, Some(0.5));
        assert_eq!(output[0].empirical_power, None);
        assert_eq!(output[0].regime, HypothesisRegime::TrueNull);
    }

    #[test]
    fn alternative_dgp_rejection_rate_is_reported_as_power() {
        let records = vec![
            record("1", HypothesisRegime::Alternative, RunStatus::Completed, Some(0.001), None, None),
            record("2", HypothesisRegime::Alternative, RunStatus::Completed, Some(0.02), None, None),
            record("3", HypothesisRegime::Alternative, RunStatus::Completed, Some(0.20), None, None),
            record("4", HypothesisRegime::Alternative, RunStatus::Completed, Some(0.90), None, None),
        ];
        let output = aggregate_scorecard(&records, 0.05).unwrap();
        assert_eq!(output[0].rejected_replicates, 2);
        assert_eq!(output[0].rejection_rate, Some(0.5));
        assert_eq!(output[0].empirical_size, None);
        assert_eq!(output[0].empirical_power, Some(0.5));
        assert_eq!(output[0].regime, HypothesisRegime::Alternative);
    }

    #[test]
    fn null_and_alternative_regimes_are_never_pooled() {
        let records = vec![
            record("1", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.01), None, None),
            record("1", HypothesisRegime::Alternative, RunStatus::Completed, Some(0.20), None, None),
        ];
        let output = aggregate_scorecard(&records, 0.05).unwrap();
        assert_eq!(output.len(), 2);
        assert_ne!(output[0].regime, output[1].regime);
    }

    #[test]
    fn missing_test_outputs_produce_conservative_rejection_bounds() {
        let records = vec![
            record("1", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.01), None, None),
            record("2", HypothesisRegime::TrueNull, RunStatus::Failed, None, None, None),
            record("3", HypothesisRegime::TrueNull, RunStatus::Completed, None, None, None),
        ];
        let output = aggregate_scorecard(&records, 0.05).unwrap();
        assert_eq!(output[0].attempted_replicates, 3);
        assert_eq!(output[0].failed_replicates, 1);
        assert_eq!(output[0].missing_p_value_replicates, 2);
        assert_eq!(output[0].rejection_rate_missing_all_nonreject, Some(1.0 / 3.0));
        assert_eq!(output[0].rejection_rate_missing_all_reject, Some(1.0));
    }

    #[test]
    fn missing_confidence_sets_produce_coverage_sensitivity_bounds() {
        let records = vec![
            record("1", HypothesisRegime::TrueNull, RunStatus::Completed, None, None, Some(bounded(1.0, 3.0))),
            record("2", HypothesisRegime::TrueNull, RunStatus::Completed, None, None, None),
            record("3", HypothesisRegime::TrueNull, RunStatus::Failed, None, None, None),
        ];
        let output = aggregate_scorecard(&records, 0.05).unwrap();
        assert_eq!(output[0].confidence_set_replicates, 1);
        assert_eq!(output[0].covered_replicates, 1);
        assert_eq!(output[0].coverage_rate, Some(1.0));
        assert_eq!(output[0].coverage_rate_missing_uncovered, Some(1.0 / 3.0));
        assert_eq!(output[0].coverage_rate_missing_covered, Some(1.0));
    }

    #[test]
    fn full_union_is_used_for_coverage() {
        let set = ConfidenceSet {
            intervals: vec![interval(Some(0.0), Some(1.0)), interval(Some(3.0), Some(4.0))],
        };
        assert!(set.contains(3.5));
        assert!(!set.contains(2.0));
    }

    #[test]
    fn point_estimate_bias_and_rmse_are_reported_separately() {
        let records = vec![
            record("1", HypothesisRegime::TrueNull, RunStatus::Completed, None, Some(1.0), None),
            record("2", HypothesisRegime::TrueNull, RunStatus::Completed, None, Some(3.0), None),
        ];
        let output = aggregate_scorecard(&records, 0.05).unwrap();
        assert_eq!(output[0].mean_estimate, Some(2.0));
        assert_eq!(output[0].bias, Some(0.0));
        assert_eq!(output[0].rmse, Some(1.0));
    }

    #[test]
    fn unbounded_sets_are_counted_but_never_given_zero_width() {
        let records = vec![
            record("1", HypothesisRegime::TrueNull, RunStatus::Completed, None, None, Some(ConfidenceSet::whole_line())),
            record("2", HypothesisRegime::TrueNull, RunStatus::Completed, None, None, Some(bounded(0.0, 1.0))),
        ];
        let output = aggregate_scorecard(&records, 0.05).unwrap();
        assert_eq!(output[0].topology_counts.whole_line, 1);
        assert_eq!(output[0].topology_counts.bounded, 1);
        assert_eq!(output[0].unbounded_confidence_set_replicates, 1);
        assert_eq!(output[0].finite_width_replicates, 1);
        assert_eq!(output[0].median_finite_set_width, Some(1.0));
    }

    #[test]
    fn pvalue_histogram_has_ten_equal_bins_and_places_one_in_last() {
        let mut records = Vec::new();
        for i in 0..10 {
            records.push(record(
                &i.to_string(),
                HypothesisRegime::TrueNull,
                RunStatus::Completed,
                Some(i as f64 / 10.0),
                None,
                None,
            ));
        }
        records.push(record("10", HypothesisRegime::TrueNull, RunStatus::Completed, Some(1.0), None, None));
        let output = aggregate_scorecard(&records, 0.05).unwrap();
        assert_eq!(output[0].p_value_histogram_10.iter().sum::<usize>(), 11);
        assert_eq!(output[0].p_value_histogram_10[0], 1);
        assert_eq!(output[0].p_value_histogram_10[9], 2);
    }

    #[test]
    fn failure_and_not_applicable_counts_are_separate() {
        let records = vec![
            record("1", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.4), None, None),
            record("2", HypothesisRegime::TrueNull, RunStatus::Failed, None, None, None),
            record("3", HypothesisRegime::TrueNull, RunStatus::NotApplicable, None, None, None),
        ];
        let output = aggregate_scorecard(&records, 0.05).unwrap();
        assert_eq!(output[0].attempted_replicates, 2);
        assert_eq!(output[0].completed_replicates, 1);
        assert_eq!(output[0].failed_replicates, 1);
        assert_eq!(output[0].not_applicable_replicates, 1);
        assert_eq!(output[0].failure_rate, Some(0.5));
    }

    #[test]
    fn duplicate_replication_ids_within_group_fail_closed() {
        let records = vec![
            record("same", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.1), None, None),
            record("same", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.2), None, None),
        ];
        assert!(matches!(
            aggregate_scorecard(&records, 0.05),
            Err(ScorecardError::DuplicateReplicationId { .. })
        ));
    }

    #[test]
    fn true_beta_must_be_constant_within_scenario_method_regime() {
        let mut second = record("2", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.2), None, None);
        second.true_beta = 4.0;
        let records = vec![
            record("1", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.1), None, None),
            second,
        ];
        assert!(matches!(
            aggregate_scorecard(&records, 0.05),
            Err(ScorecardError::InconsistentTrueBeta { .. })
        ));
    }

    #[test]
    fn completed_run_without_any_metric_fails_closed() {
        let run = record("1", HypothesisRegime::TrueNull, RunStatus::Completed, None, None, None);
        assert_eq!(run.validate(), Err(ScorecardError::CompletedWithoutMetrics));
    }

    #[test]
    fn failed_run_cannot_smuggle_inferential_metrics() {
        let run = record("1", HypothesisRegime::TrueNull, RunStatus::Failed, Some(0.1), None, None);
        assert_eq!(run.validate(), Err(ScorecardError::NonCompletedRunHasMetrics));
    }

    #[test]
    fn invalid_p_value_fails_closed() {
        let run = record("1", HypothesisRegime::TrueNull, RunStatus::Completed, Some(1.1), None, None);
        assert!(matches!(run.validate(), Err(ScorecardError::InvalidPValue { .. })));
    }

    #[test]
    fn empty_input_returns_empty_scorecard() {
        assert!(aggregate_scorecard(&[], 0.05).unwrap().is_empty());
    }

    #[test]
    fn invalid_alpha_and_wilson_critical_value_fail_closed() {
        assert!(matches!(aggregate_scorecard(&[], 0.0), Err(ScorecardError::InvalidAlpha { .. })));
        assert!(matches!(wilson_interval(1, 2, 0.0), Err(ScorecardError::InvalidCriticalValue { .. })));
    }

    #[test]
    fn aggregation_is_independent_of_input_row_order() {
        let a = record("1", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.01), Some(1.0), Some(bounded(1.0, 2.0)));
        let b = record("2", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.20), Some(3.0), Some(bounded(1.0, 4.0)));
        let left = aggregate_scorecard(&[a.clone(), b.clone()], 0.05).unwrap();
        let right = aggregate_scorecard(&[b, a], 0.05).unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn runtime_and_width_quantiles_are_reported_on_finite_observations() {
        let mut a = record("1", HypothesisRegime::TrueNull, RunStatus::Completed, None, None, Some(bounded(0.0, 1.0)));
        a.runtime_micros = Some(100);
        let mut b = record("2", HypothesisRegime::TrueNull, RunStatus::Completed, None, None, Some(bounded(0.0, 3.0)));
        b.runtime_micros = Some(300);
        let output = aggregate_scorecard(&[a, b], 0.05).unwrap();
        assert_eq!(output[0].median_finite_set_width, Some(2.0));
        assert_eq!(output[0].p90_finite_set_width, Some(2.8));
        assert_eq!(output[0].median_runtime_micros, Some(200.0));
        assert_eq!(output[0].p95_runtime_micros, Some(290.0));
    }

    #[test]
    fn no_test_results_has_no_rejection_rate_or_interval() {
        let records = vec![record("1", HypothesisRegime::TrueNull, RunStatus::Completed, None, Some(1.0), None)];
        let output = aggregate_scorecard(&records, 0.05).unwrap();
        assert_eq!(output[0].p_value_replicates, 0);
        assert_eq!(output[0].rejection_rate, None);
        assert_eq!(output[0].rejection_wilson_95, None);
    }

    #[test]
    fn groups_are_sorted_deterministically_by_scenario_method_and_regime() {
        let mut other = record("1", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.1), None, None);
        other.method_id = "method-b".to_owned();
        let original = record("1", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.1), None, None);
        let output = aggregate_scorecard(&[other, original], 0.05).unwrap();
        assert_eq!(output[0].method_id, "method-a");
        assert_eq!(output[1].method_id, "method-b");
    }

    #[test]
    fn disconnected_set_width_sums_all_components_not_convex_hull() {
        let set = ConfidenceSet {
            intervals: vec![interval(Some(0.0), Some(1.0)), interval(Some(4.0), Some(6.0))],
        };
        set.validate().unwrap();
        assert_eq!(set.finite_width(), Some(3.0));
    }

    #[test]
    fn invalid_non_finite_true_parameter_fails_closed() {
        let mut run = record("1", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.1), None, None);
        run.true_beta = f64::INFINITY;
        assert!(matches!(run.validate(), Err(ScorecardError::NonFiniteParameter { field: "true_beta" })));
    }

    #[test]
    fn invalid_non_finite_estimate_fails_closed() {
        let mut run = record("1", HypothesisRegime::TrueNull, RunStatus::Completed, None, Some(f64::NAN), None);
        assert!(matches!(run.validate(), Err(ScorecardError::NonFiniteParameter { field: "point_estimate" })));
    }

    #[test]
    fn bounded_disconnected_set_is_not_counted_as_unbounded() {
        let set = ConfidenceSet {
            intervals: vec![
                interval(Some(0.0), Some(1.0)),
                interval(Some(3.0), Some(4.0)),
            ],
        };
        set.validate().unwrap();
        assert_eq!(set.topology(), ConfidenceSetTopology::Disconnected);
        assert_eq!(set.finite_width(), Some(2.0));
        let run = record(
            "1",
            HypothesisRegime::TrueNull,
            RunStatus::Completed,
            None,
            None,
            Some(set),
        );
        let summary = aggregate_scorecard(&[run], 0.05).unwrap();
        assert_eq!(summary[0].topology_counts.disconnected, 1);
        assert_eq!(summary[0].unbounded_confidence_set_replicates, 0);
    }

    #[test]
    fn disconnected_set_with_an_unbounded_component_is_counted_as_unbounded() {
        let set = ConfidenceSet {
            intervals: vec![
                interval(None, Some(1.0)),
                interval(Some(3.0), Some(4.0)),
            ],
        };
        set.validate().unwrap();
        assert_eq!(set.topology(), ConfidenceSetTopology::Disconnected);
        assert_eq!(set.finite_width(), None);
        let run = record(
            "1",
            HypothesisRegime::TrueNull,
            RunStatus::Completed,
            None,
            None,
            Some(set),
        );
        let summary = aggregate_scorecard(&[run], 0.05).unwrap();
        assert_eq!(summary[0].topology_counts.disconnected, 1);
        assert_eq!(summary[0].unbounded_confidence_set_replicates, 1);
        assert_eq!(summary[0].finite_width_replicates, 0);
    }

    #[test]
    fn no_confidence_set_outputs_do_not_create_artificial_zero_to_one_coverage_bounds() {
        let run = record(
            "1",
            HypothesisRegime::TrueNull,
            RunStatus::Completed,
            Some(0.5),
            None,
            None,
        );
        let summary = aggregate_scorecard(&[run], 0.05).unwrap();
        assert_eq!(summary[0].coverage_rate, None);
        assert_eq!(summary[0].coverage_rate_missing_uncovered, None);
        assert_eq!(summary[0].coverage_rate_missing_covered, None);
    }


    #[test]
    fn true_null_requires_dgp_truth_equal_to_tested_null() {
        let mut run = record(
            "1", HypothesisRegime::TrueNull, RunStatus::Completed,
            Some(0.1), None, None,
        );
        run.tested_null_beta = 0.0;
        assert!(matches!(
            run.validate(),
            Err(ScorecardError::RegimeTruthMismatch { .. })
        ));
    }

    #[test]
    fn alternative_requires_dgp_truth_different_from_tested_null() {
        let mut run = record(
            "1", HypothesisRegime::Alternative, RunStatus::Completed,
            Some(0.1), None, None,
        );
        run.tested_null_beta = run.true_beta;
        assert!(matches!(
            run.validate(),
            Err(ScorecardError::RegimeTruthMismatch { .. })
        ));
    }

    #[test]
    fn tested_null_beta_must_be_consistent_within_a_group() {
        let first = record(
            "1", HypothesisRegime::Alternative, RunStatus::Completed,
            Some(0.1), None, None,
        );
        let mut second = record(
            "2", HypothesisRegime::Alternative, RunStatus::Completed,
            Some(0.2), None, None,
        );
        second.tested_null_beta = 1.0;
        assert!(matches!(
            aggregate_scorecard(&[first, second], 0.05),
            Err(ScorecardError::InconsistentTestedNullBeta { .. })
        ));
    }

    #[test]
    fn non_finite_tested_null_beta_fails_closed() {
        let mut run = record(
            "1", HypothesisRegime::TrueNull, RunStatus::Completed,
            Some(0.1), None, None,
        );
        run.tested_null_beta = f64::NAN;
        assert!(matches!(
            run.validate(),
            Err(ScorecardError::NonFiniteParameter { field: "tested_null_beta" })
        ));
    }


    #[test]
    fn json_report_contains_size_power_and_topology_separately() {
        let null_run = record(
            "1",
            HypothesisRegime::TrueNull,
            RunStatus::Completed,
            Some(0.01),
            Some(2.1),
            Some(bounded(1.0, 3.0)),
        );
        let alt_run = record(
            "2",
            HypothesisRegime::Alternative,
            RunStatus::Completed,
            Some(0.2),
            Some(2.4),
            Some(ConfidenceSet::whole_line()),
        );
        let report = aggregate_scorecard_report(&[null_run, alt_run], 0.05).unwrap();
        let json = report.to_json();
        assert!(json.starts_with("{\"schema_version\":\"1.0.0\""));
        assert!(json.contains("\"empirical_size\":"));
        assert!(json.contains("\"empirical_power\":"));
        assert!(json.contains("\"tested_null_beta\":"));
        assert!(json.contains("\"unbounded_confidence_set_replicates\":"));
        assert!(json.ends_with("]\n"));
    }

    #[test]
    fn json_report_escapes_scenario_and_method_identifiers() {
        let mut run = record(
            "1",
            HypothesisRegime::TrueNull,
            RunStatus::Completed,
            Some(0.5),
            None,
            None,
        );
        run.scenario_id = "scenario \"quoted\"".to_owned();
        run.method_id = "method\\path".to_owned();
        let json = aggregate_scorecard_report(&[run], 0.05).unwrap().to_json();
        assert!(json.contains("scenario \\\"quoted\\\""));
        assert!(json.contains("method\\\\path"));
    }


    #[test]
    fn integrated_null_and_alternative_known_answers_preserve_denominators() {
        let mut records = Vec::new();

        let mut n1 = record("n1", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.01), Some(0.05), Some(bounded(-0.1, 0.1)));
        n1.true_beta = 0.0; n1.tested_null_beta = 0.0; n1.runtime_micros = Some(100);
        let mut n2 = record("n2", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.10), Some(-0.03), Some(bounded(0.05, 0.2)));
        n2.true_beta = 0.0; n2.tested_null_beta = 0.0; n2.runtime_micros = Some(200);
        let mut n3 = record("n3", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.20), Some(0.02), Some(bounded(-0.3, 0.3)));
        n3.true_beta = 0.0; n3.tested_null_beta = 0.0; n3.runtime_micros = Some(300);
        let mut n4 = record("n4", HypothesisRegime::TrueNull, RunStatus::Completed, Some(0.03), Some(0.20), Some(ConfidenceSet::whole_line()));
        n4.true_beta = 0.0; n4.tested_null_beta = 0.0; n4.runtime_micros = Some(400);
        let mut n5 = record("n5", HypothesisRegime::TrueNull, RunStatus::Failed, None, None, None);
        n5.true_beta = 0.0; n5.tested_null_beta = 0.0; n5.runtime_micros = Some(500);
        let mut n6 = record("n6", HypothesisRegime::TrueNull, RunStatus::NotApplicable, None, None, None);
        n6.true_beta = 0.0; n6.tested_null_beta = 0.0; n6.runtime_micros = None;

        let mut a1 = record("a1", HypothesisRegime::Alternative, RunStatus::Completed, Some(0.001), Some(0.9), Some(bounded(-0.1, 0.8)));
        a1.true_beta = 1.0; a1.tested_null_beta = 0.0; a1.runtime_micros = Some(120);
        let mut a2 = record("a2", HypothesisRegime::Alternative, RunStatus::Completed, Some(0.01), Some(1.1), Some(bounded(0.7, 1.3)));
        a2.true_beta = 1.0; a2.tested_null_beta = 0.0; a2.runtime_micros = Some(180);
        let mut a3 = record("a3", HypothesisRegime::Alternative, RunStatus::Completed, Some(0.04), Some(0.8), Some(bounded(0.5, 1.2)));
        a3.true_beta = 1.0; a3.tested_null_beta = 0.0; a3.runtime_micros = Some(220);
        let mut a4 = record("a4", HypothesisRegime::Alternative, RunStatus::Completed, Some(0.20), Some(1.0), Some(ConfidenceSet::whole_line()));
        a4.true_beta = 1.0; a4.tested_null_beta = 0.0; a4.runtime_micros = Some(400);
        let mut a5 = record("a5", HypothesisRegime::Alternative, RunStatus::Completed, Some(0.60), Some(1.2), Some(bounded(0.8, 0.95)));
        a5.true_beta = 1.0; a5.tested_null_beta = 0.0; a5.runtime_micros = Some(600);

        records.extend([n1, n2, n3, n4, n5, n6, a1, a2, a3, a4, a5]);
        let summary = aggregate_scorecard(&records, 0.05).unwrap();
        assert_eq!(summary.len(), 2);

        let null = summary.iter().find(|group| group.regime == HypothesisRegime::TrueNull).unwrap();
        assert_eq!(null.attempted_replicates, 5);
        assert_eq!(null.completed_replicates, 4);
        assert_eq!(null.failed_replicates, 1);
        assert_eq!(null.not_applicable_replicates, 1);
        assert_eq!(null.empirical_size, Some(0.5));
        assert_eq!(null.rejection_rate_missing_all_nonreject, Some(0.4));
        assert_eq!(null.rejection_rate_missing_all_reject, Some(0.6));
        assert_eq!(null.coverage_rate, Some(0.75));
        assert_eq!(null.coverage_rate_missing_uncovered, Some(0.6));
        assert_eq!(null.coverage_rate_missing_covered, Some(0.8));
        assert_eq!(null.topology_counts.bounded, 3);
        assert_eq!(null.topology_counts.whole_line, 1);
        assert_eq!(null.unbounded_confidence_set_replicates, 1);
        assert_eq!(null.median_finite_set_width, Some(0.2));
        assert!((null.p90_finite_set_width.unwrap() - 0.56).abs() < 1.0e-12);
        assert!((null.bias.unwrap() - 0.06).abs() < 1.0e-12);
        assert!((null.rmse.unwrap() - 0.104642247862).abs() < 1.0e-9);

        let alternative = summary.iter().find(|group| group.regime == HypothesisRegime::Alternative).unwrap();
        assert_eq!(alternative.attempted_replicates, 5);
        assert_eq!(alternative.empirical_power, Some(0.6));
        assert_eq!(alternative.coverage_rate, Some(0.6));
        assert_eq!(alternative.topology_counts.bounded, 4);
        assert_eq!(alternative.topology_counts.whole_line, 1);
        assert_eq!(alternative.unbounded_confidence_set_replicates, 1);
        assert_eq!(alternative.median_finite_set_width, Some(0.65));
        assert!((alternative.p90_finite_set_width.unwrap() - 0.84).abs() < 1.0e-12);
        assert_eq!(alternative.bias, Some(0.0));
        assert!((alternative.rmse.unwrap() - 0.141421356237).abs() < 1.0e-9);
    }

}
