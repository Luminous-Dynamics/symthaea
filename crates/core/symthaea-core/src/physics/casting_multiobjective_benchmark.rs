// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Test-only two-objective Pareto archive and hypervolume metric for CASTING-002D.
//!
//! Both objectives are minimized. The reference point must be frozen before method
//! comparison and be strictly worse than every feasible point in each objective.
//! Infeasible and failed trials count toward attempt/cost metrics, but do not enter
//! the feasible Pareto front. This module does not model physical casting outcomes.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParetoTrialStatus {
    Feasible,
    Infeasible,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ParetoTrial {
    trial_id: u64,
    status: ParetoTrialStatus,
    objectives: Option<[f64; 2]>,
    cost_units: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ParetoPoint {
    trial_id: u64,
    objectives: [f64; 2],
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HypervolumeTraceEntry {
    trial_id: u64,
    status: ParetoTrialStatus,
    cumulative_cost_units: u64,
    hypervolume: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParetoMetricError {
    InvalidReferencePoint,
    DuplicateTrialId,
    InvalidCost,
    CostOverflow,
    InconsistentOutcome,
    NonFiniteObjective,
    OutsideReferenceBox,
    InvalidHypervolumeTarget,
    EmptyApproximationFront,
    EmptyReferenceFront,
    TooFewPointsForSpacing,
    NonFiniteMetricResult,
    InvalidDecisionDimension,
    NonFiniteDecisionVariable,
    DecisionOutOfDomain,
    InvalidReferenceFrontResolution,
    InsufficientFrontForGap,
    EmptyPredictedCandidateSet,
    DuplicateCandidateId,
    InvalidObjectiveScale,
    InvalidConfidenceMultiplier,
    InvalidPredictionUncertainty,
    NoEpsilonFeasibleCandidate,
    InvalidParetoFront,
    EmptyCalibrationSet,
    EmptyAssessmentSet,
    InvalidMiscoverage,
    InsufficientCalibrationSamples,
    InvalidCalibrationObservation,
    InvalidPredictiveScale,
    InvalidConformalMultiplier,
    NonFiniteCalibrationMetric,
    InvalidImportanceWeight,
    ZeroTotalCalibrationWeight,
    InsufficientWeightedCalibrationSupport,
    NonFiniteWeightedConformalMetric,
    EmptyCoverageAuditSet,
    InvalidCoverageAuditParameter,
    InvalidResidualScore,
    MissingSelectedCoverage,
    MissingUnselectedCoverage,
    InsufficientCoverageGroupSample,
}

impl fmt::Display for ParetoMetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidReferencePoint => f.write_str("hypervolume reference point must be finite"),
            Self::DuplicateTrialId => f.write_str("multi-objective trial ID was already recorded"),
            Self::InvalidCost => f.write_str("each attempted trial must have positive cost"),
            Self::CostOverflow => f.write_str("cumulative multi-objective trial cost overflowed"),
            Self::InconsistentOutcome => {
                f.write_str("feasible trials require two objectives; failed/infeasible trials require none")
            }
            Self::NonFiniteObjective => f.write_str("feasible objective values must be finite"),
            Self::OutsideReferenceBox => {
                f.write_str("for minimization, each feasible objective must be strictly better than the frozen reference point")
            }
            Self::InvalidHypervolumeTarget => {
                f.write_str("hypervolume target must be finite and strictly positive")
            }
            Self::EmptyApproximationFront => f.write_str("approximation front must not be empty"),
            Self::EmptyReferenceFront => f.write_str("reference front must not be empty"),
            Self::TooFewPointsForSpacing => {
                f.write_str("spacing requires at least two approximation points")
            }
            Self::NonFiniteMetricResult => f.write_str("front metric overflowed or produced a non-finite result"),
            Self::InvalidDecisionDimension => {
                f.write_str("ZDT1/ZDT2/ZDT3 require 30 variables; ZDT4 requires 10")
            }
            Self::NonFiniteDecisionVariable => f.write_str("ZDT decision variables must be finite"),
            Self::DecisionOutOfDomain => {
                f.write_str("ZDT decision variable is outside the problem's declared bounds")
            }
            Self::InvalidReferenceFrontResolution => {
                f.write_str("each reference-front segment requires at least two samples")
            }
            Self::InsufficientFrontForGap => {
                f.write_str("gap targeting requires at least two distinct front points")
            }
            Self::EmptyPredictedCandidateSet => {
                f.write_str("epsilon proposal needs at least one candidate prediction")
            }
            Self::DuplicateCandidateId => f.write_str("candidate prediction IDs must be unique"),
            Self::InvalidObjectiveScale => {
                f.write_str("objective normalization scales must be finite and positive")
            }
            Self::InvalidConfidenceMultiplier => {
                f.write_str("objective confidence multipliers must be finite and non-negative")
            }
            Self::InvalidPredictionUncertainty => {
                f.write_str("predicted means and standard deviations must be finite and standard deviations non-negative")
            }
            Self::NoEpsilonFeasibleCandidate => {
                f.write_str("no predicted candidate satisfies the upper-confidence epsilon constraint")
            }
            Self::InvalidParetoFront => {
                f.write_str("front must contain only mutually non-dominated objective points")
            }
            Self::EmptyCalibrationSet => f.write_str("conformal calibration set must not be empty"),
            Self::EmptyAssessmentSet => f.write_str("holdout assessment set must not be empty"),
            Self::InvalidMiscoverage => f.write_str("conformal miscoverage alpha must be finite and in (0, 1)"),
            Self::InsufficientCalibrationSamples => f.write_str("calibration set is too small for the requested finite-sample quantile"),
            Self::InvalidCalibrationObservation => f.write_str("calibration predictions and observed values must be finite"),
            Self::InvalidPredictiveScale => f.write_str("predictive standard deviation must be finite and strictly positive"),
            Self::InvalidConformalMultiplier => f.write_str("conformal multiplier must be finite and non-negative"),
            Self::NonFiniteCalibrationMetric => f.write_str("conformal calibration score or assessment metric is non-finite"),
            Self::InvalidImportanceWeight => f.write_str("importance weights must be finite and non-negative; target-point weight must be positive"),
            Self::ZeroTotalCalibrationWeight => f.write_str("weighted calibration requires positive total calibration weight"),
            Self::InsufficientWeightedCalibrationSupport => f.write_str("test-point weight leaves insufficient finite calibration mass for the requested weighted conformal quantile"),
            Self::NonFiniteWeightedConformalMetric => f.write_str("weighted conformal normalization, effective sample size, or quantile mass is non-finite"),
            Self::EmptyCoverageAuditSet => f.write_str("post-selection coverage audit requires held-out observations"),
            Self::InvalidCoverageAuditParameter => f.write_str("coverage multiplier and nominal level must be valid and minimum group size positive"),
            Self::InvalidResidualScore => f.write_str("standardized residual scores must be finite and non-negative"),
            Self::MissingSelectedCoverage => f.write_str("post-selection audit requires at least one optimizer-selected observation"),
            Self::MissingUnselectedCoverage => f.write_str("post-selection audit requires at least one unselected observation"),
            Self::InsufficientCoverageGroupSample => f.write_str("at least one regional group has fewer samples than the frozen minimum"),
        }
    }
}

impl std::error::Error for ParetoMetricError {}

#[derive(Debug, Clone, PartialEq)]
struct ParetoArchive {
    reference_point: [f64; 2],
    trials: Vec<ParetoTrial>,
    front: Vec<ParetoPoint>,
    trace: Vec<HypervolumeTraceEntry>,
    total_cost_units: u64,
    feasible_count: usize,
    infeasible_count: usize,
    failed_count: usize,
}

impl ParetoArchive {
    fn new(reference_point: [f64; 2]) -> Result<Self, ParetoMetricError> {
        if reference_point.iter().any(|value| !value.is_finite()) {
            return Err(ParetoMetricError::InvalidReferencePoint);
        }
        Ok(Self {
            reference_point,
            trials: Vec::new(),
            front: Vec::new(),
            trace: Vec::new(),
            total_cost_units: 0,
            feasible_count: 0,
            infeasible_count: 0,
            failed_count: 0,
        })
    }

    /// Record one complete attempt atomically. Validation happens before counters,
    /// costs, the Pareto front, or the transcript are mutated.
    fn observe(&mut self, trial: ParetoTrial) -> Result<(), ParetoMetricError> {
        if self.trials.iter().any(|existing| existing.trial_id == trial.trial_id) {
            return Err(ParetoMetricError::DuplicateTrialId);
        }
        if trial.cost_units == 0 {
            return Err(ParetoMetricError::InvalidCost);
        }
        let next_total_cost = self
            .total_cost_units
            .checked_add(trial.cost_units)
            .ok_or(ParetoMetricError::CostOverflow)?;

        match (trial.status, trial.objectives) {
            (ParetoTrialStatus::Feasible, Some(objectives)) => {
                if objectives.iter().any(|value| !value.is_finite()) {
                    return Err(ParetoMetricError::NonFiniteObjective);
                }
                if objectives[0] >= self.reference_point[0]
                    || objectives[1] >= self.reference_point[1]
                {
                    return Err(ParetoMetricError::OutsideReferenceBox);
                }
            }
            (ParetoTrialStatus::Feasible, None)
            | (ParetoTrialStatus::Infeasible, Some(_))
            | (ParetoTrialStatus::Failed, Some(_)) => {
                return Err(ParetoMetricError::InconsistentOutcome);
            }
            (ParetoTrialStatus::Infeasible, None) | (ParetoTrialStatus::Failed, None) => {}
        }

        self.total_cost_units = next_total_cost;
        self.trials.push(trial);
        match trial.status {
            ParetoTrialStatus::Feasible => {
                self.feasible_count += 1;
                self.insert_feasible_point(ParetoPoint {
                    trial_id: trial.trial_id,
                    objectives: trial.objectives.expect("validated feasible trial has objectives"),
                });
            }
            ParetoTrialStatus::Infeasible => self.infeasible_count += 1,
            ParetoTrialStatus::Failed => self.failed_count += 1,
        }

        self.trace.push(HypervolumeTraceEntry {
            trial_id: trial.trial_id,
            status: trial.status,
            cumulative_cost_units: self.total_cost_units,
            hypervolume: self.hypervolume(),
        });
        Ok(())
    }

    fn insert_feasible_point(&mut self, candidate: ParetoPoint) {
        // Equal objective vectors describe the same point in objective space. Keep the
        // lowest ID as deterministic representative so insertion order cannot change the front.
        if let Some(existing) = self
            .front
            .iter()
            .find(|point| point.objectives == candidate.objectives)
            .copied()
        {
            if existing.trial_id <= candidate.trial_id {
                return;
            }
            self.front
                .retain(|point| point.objectives != candidate.objectives);
        }

        if self
            .front
            .iter()
            .any(|existing| dominates(existing.objectives, candidate.objectives))
        {
            return;
        }

        self.front
            .retain(|existing| !dominates(candidate.objectives, existing.objectives));
        self.front.push(candidate);
        self.front.sort_by(|left, right| {
            left.objectives[0]
                .total_cmp(&right.objectives[0])
                .then_with(|| left.objectives[1].total_cmp(&right.objectives[1]))
                .then_with(|| left.trial_id.cmp(&right.trial_id))
        });
    }

    /// Exact 2-D hypervolume for minimization and the frozen upper-right reference point.
    /// A slice sweep is exact for two objectives; do not silently generalize it to 3+ objectives.
    fn hypervolume(&self) -> f64 {
        if self.front.is_empty() {
            return 0.0;
        }
        let mut area = 0.0;
        for (index, point) in self.front.iter().enumerate() {
            let next_x = self
                .front
                .get(index + 1)
                .map(|next| next.objectives[0])
                .unwrap_or(self.reference_point[0]);
            let width = (next_x - point.objectives[0]).max(0.0);
            let height = self.reference_point[1] - point.objectives[1];
            area += width * height;
        }
        area
    }

    fn cost_to_hypervolume_target(
        &self,
        target: f64,
    ) -> Result<Option<u64>, ParetoMetricError> {
        if !target.is_finite() || target <= 0.0 {
            return Err(ParetoMetricError::InvalidHypervolumeTarget);
        }
        Ok(self
            .trace
            .iter()
            .find(|entry| entry.hypervolume >= target)
            .map(|entry| entry.cumulative_cost_units))
    }

    fn front(&self) -> &[ParetoPoint] {
        &self.front
    }

    fn trace(&self) -> &[HypervolumeTraceEntry] {
        &self.trace
    }

    fn total_cost_units(&self) -> u64 {
        self.total_cost_units
    }

    fn attempt_count(&self) -> usize {
        self.trials.len()
    }

    fn status_counts(&self) -> (usize, usize, usize) {
        (self.feasible_count, self.infeasible_count, self.failed_count)
    }
}

fn dominates(left: [f64; 2], right: [f64; 2]) -> bool {
    left[0] <= right[0]
        && left[1] <= right[1]
        && (left[0] < right[0] || left[1] < right[1])
}

/// Validate objective vectors used by cross-front metrics.
/// All front values must already use the same objective directions and normalization.
fn validate_metric_front(points: &[[f64; 2]]) -> Result<(), ParetoMetricError> {
    if points.iter().flatten().any(|value| !value.is_finite()) {
        return Err(ParetoMetricError::NonFiniteObjective);
    }
    Ok(())
}

/// Additive epsilon indicator for minimization:
/// max over each reference point z of min over approximation points a of max_i(a_i - z_i).
/// Lower is better; a negative value means the approximation dominates every reference point.
fn additive_epsilon_indicator(
    approximation: &[[f64; 2]],
    reference: &[[f64; 2]],
) -> Result<f64, ParetoMetricError> {
    if approximation.is_empty() {
        return Err(ParetoMetricError::EmptyApproximationFront);
    }
    if reference.is_empty() {
        return Err(ParetoMetricError::EmptyReferenceFront);
    }
    validate_metric_front(approximation)?;
    validate_metric_front(reference)?;

    let mut worst_reference_gap = f64::NEG_INFINITY;
    for target in reference {
        let mut best_approximation_gap = f64::INFINITY;
        for candidate in approximation {
            let delta_0 = candidate[0] - target[0];
            let delta_1 = candidate[1] - target[1];
            if !delta_0.is_finite() || !delta_1.is_finite() {
                return Err(ParetoMetricError::NonFiniteMetricResult);
            }
            best_approximation_gap = best_approximation_gap.min(delta_0.max(delta_1));
        }
        worst_reference_gap = worst_reference_gap.max(best_approximation_gap);
    }
    if worst_reference_gap.is_finite() {
        Ok(worst_reference_gap)
    } else {
        Err(ParetoMetricError::NonFiniteMetricResult)
    }
}

/// Inverted generational distance plus (IGD+) for minimization.
/// For each reference point z, measure only the positive, worse-than-reference
/// components max(a_i - z_i, 0) of the nearest approximation point a; return their mean.
/// Lower is better. This implementation is 2-D and expects normalized, comparable objectives.
fn inverted_generational_distance_plus(
    approximation: &[[f64; 2]],
    reference: &[[f64; 2]],
) -> Result<f64, ParetoMetricError> {
    if approximation.is_empty() {
        return Err(ParetoMetricError::EmptyApproximationFront);
    }
    if reference.is_empty() {
        return Err(ParetoMetricError::EmptyReferenceFront);
    }
    validate_metric_front(approximation)?;
    validate_metric_front(reference)?;

    // Incremental mean avoids overflowing a sum of many finite distances.
    let mut mean_distance = 0.0;
    for (index, target) in reference.iter().enumerate() {
        let mut best_distance = f64::INFINITY;
        for candidate in approximation {
            let raw_0 = candidate[0] - target[0];
            let raw_1 = candidate[1] - target[1];
            if !raw_0.is_finite() || !raw_1.is_finite() {
                return Err(ParetoMetricError::NonFiniteMetricResult);
            }
            let distance = raw_0.max(0.0).hypot(raw_1.max(0.0));
            if !distance.is_finite() {
                return Err(ParetoMetricError::NonFiniteMetricResult);
            }
            best_distance = best_distance.min(distance);
        }
        mean_distance += (best_distance - mean_distance) / (index + 1) as f64;
        if !mean_distance.is_finite() {
            return Err(ParetoMetricError::NonFiniteMetricResult);
        }
    }
    Ok(mean_distance)
}

/// Sample standard deviation of each point's nearest-neighbour Euclidean distance.
/// Lower values indicate more even spacing, not a better or more complete Pareto front.
/// Input points must be distinct or duplicate points intentionally contribute zero spacing.
fn nearest_neighbour_spacing(
    approximation: &[[f64; 2]],
) -> Result<f64, ParetoMetricError> {
    if approximation.len() < 2 {
        return Err(ParetoMetricError::TooFewPointsForSpacing);
    }
    validate_metric_front(approximation)?;

    let mut nearest_distances = Vec::with_capacity(approximation.len());
    for (index, point) in approximation.iter().enumerate() {
        let mut nearest = f64::INFINITY;
        for (other_index, other) in approximation.iter().enumerate() {
            if index == other_index {
                continue;
            }
            let delta_0 = point[0] - other[0];
            let delta_1 = point[1] - other[1];
            if !delta_0.is_finite() || !delta_1.is_finite() {
                return Err(ParetoMetricError::NonFiniteMetricResult);
            }
            let distance = delta_0.hypot(delta_1);
            if !distance.is_finite() {
                return Err(ParetoMetricError::NonFiniteMetricResult);
            }
            nearest = nearest.min(distance);
        }
        nearest_distances.push(nearest);
    }

    let mut mean = 0.0;
    let mut sum_squared_deviations = 0.0;
    for (index, distance) in nearest_distances.iter().enumerate() {
        let count = (index + 1) as f64;
        let delta = distance - mean;
        mean += delta / count;
        sum_squared_deviations += delta * (distance - mean);
        if !mean.is_finite() || !sum_squared_deviations.is_finite() {
            return Err(ParetoMetricError::NonFiniteMetricResult);
        }
    }
    let sample_variance = (sum_squared_deviations / (approximation.len() - 1) as f64).max(0.0);
    let spacing = sample_variance.sqrt();
    if spacing.is_finite() {
        Ok(spacing)
    } else {
        Err(ParetoMetricError::NonFiniteMetricResult)
    }
}




/// One scalar prediction/outcome pair for uncertainty calibration.
/// Fit one multiplier per objective from calibration data not used to fit the predictor.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PredictionCalibrationSample {
    predicted_mean: f64,
    predictive_std_dev: f64,
    observed_value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SplitConformalMultiplier {
    /// Target marginal miscoverage alpha (e.g. 0.1 targets 90% coverage).
    alpha: f64,
    /// Quantile of absolute standardized residuals from the held-out calibration split.
    multiplier: f64,
    calibration_count: usize,
    /// 1-based order-statistic rank ceil((n + 1) * (1 - alpha)).
    order_statistic_rank: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ConformalHoldoutAssessment {
    sample_count: usize,
    covered_count: usize,
    empirical_coverage: f64,
    mean_absolute_standardized_residual: f64,
    mean_gaussian_negative_log_likelihood: f64,
}

/// Compute a normalized absolute residual |observed - mean| / predictive_std_dev.
fn standardized_absolute_residual(
    sample: PredictionCalibrationSample,
) -> Result<f64, ParetoMetricError> {
    if !sample.predicted_mean.is_finite() || !sample.observed_value.is_finite() {
        return Err(ParetoMetricError::InvalidCalibrationObservation);
    }
    if !sample.predictive_std_dev.is_finite() || sample.predictive_std_dev <= 0.0 {
        return Err(ParetoMetricError::InvalidPredictiveScale);
    }
    let residual = sample.observed_value - sample.predicted_mean;
    if !residual.is_finite() {
        return Err(ParetoMetricError::NonFiniteCalibrationMetric);
    }
    let score = residual.abs() / sample.predictive_std_dev;
    if score.is_finite() {
        Ok(score)
    } else {
        Err(ParetoMetricError::NonFiniteCalibrationMetric)
    }
}

/// Fit split-conformal scale multiplier from normalized absolute residuals.
///
/// Under exchangeability between calibration and future cases, the order statistic
/// gives finite-sample *marginal* coverage at least 1-alpha. It does not guarantee
/// conditional coverage or validity under arbitrary manufacturing distribution shift.
fn fit_split_conformal_multiplier(
    calibration: &[PredictionCalibrationSample],
    alpha: f64,
) -> Result<SplitConformalMultiplier, ParetoMetricError> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(ParetoMetricError::InvalidMiscoverage);
    }
    if calibration.is_empty() {
        return Err(ParetoMetricError::EmptyCalibrationSet);
    }

    let mut scores = Vec::with_capacity(calibration.len());
    for sample in calibration {
        scores.push(standardized_absolute_residual(*sample)?);
    }
    scores.sort_by(f64::total_cmp);

    let rank_value = ((calibration.len() as f64 + 1.0) * (1.0 - alpha)).ceil();
    if !rank_value.is_finite()
        || rank_value < 1.0
        || rank_value > calibration.len() as f64
    {
        return Err(ParetoMetricError::InsufficientCalibrationSamples);
    }
    let order_statistic_rank = rank_value as usize;
    let multiplier = scores[order_statistic_rank - 1];
    if !multiplier.is_finite() || multiplier < 0.0 {
        return Err(ParetoMetricError::NonFiniteCalibrationMetric);
    }

    Ok(SplitConformalMultiplier {
        alpha,
        multiplier,
        calibration_count: calibration.len(),
        order_statistic_rank,
    })
}

/// Evaluate a frozen conformal multiplier on a distinct holdout set.
/// The function never changes/refits the multiplier. The caller must enforce data separation.
fn assess_conformal_holdout(
    multiplier: SplitConformalMultiplier,
    holdout: &[PredictionCalibrationSample],
) -> Result<ConformalHoldoutAssessment, ParetoMetricError> {
    if !multiplier.alpha.is_finite() || multiplier.alpha <= 0.0 || multiplier.alpha >= 1.0 {
        return Err(ParetoMetricError::InvalidMiscoverage);
    }
    if !multiplier.multiplier.is_finite() || multiplier.multiplier < 0.0 {
        return Err(ParetoMetricError::InvalidConformalMultiplier);
    }
    if holdout.is_empty() {
        return Err(ParetoMetricError::EmptyAssessmentSet);
    }

    let mut covered_count = 0usize;
    let mut mean_absolute_standardized_residual = 0.0;
    let mut mean_gaussian_negative_log_likelihood = 0.0;
    let log_sqrt_two_pi = 0.5 * (2.0 * std::f64::consts::PI).ln();

    for (index, sample) in holdout.iter().enumerate() {
        let score = standardized_absolute_residual(*sample)?;
        if score <= multiplier.multiplier {
            covered_count += 1;
        }
        let squared_score = score * score;
        let nll = log_sqrt_two_pi + sample.predictive_std_dev.ln() + 0.5 * squared_score;
        if !squared_score.is_finite() || !nll.is_finite() {
            return Err(ParetoMetricError::NonFiniteCalibrationMetric);
        }

        let count = (index + 1) as f64;
        mean_absolute_standardized_residual +=
            (score - mean_absolute_standardized_residual) / count;
        mean_gaussian_negative_log_likelihood +=
            (nll - mean_gaussian_negative_log_likelihood) / count;
        if !mean_absolute_standardized_residual.is_finite()
            || !mean_gaussian_negative_log_likelihood.is_finite()
        {
            return Err(ParetoMetricError::NonFiniteCalibrationMetric);
        }
    }

    Ok(ConformalHoldoutAssessment {
        sample_count: holdout.len(),
        covered_count,
        empirical_coverage: covered_count as f64 / holdout.len() as f64,
        mean_absolute_standardized_residual,
        mean_gaussian_negative_log_likelihood,
    })
}



/// Held-out residual score annotated with the region and whether the optimizer selected it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PostSelectionCoverageSample {
    region_id: u32,
    optimizer_selected: bool,
    standardized_residual_score: f64,
}

/// Per-region empirical coverage summary. Regions are returned in ascending region ID order.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RegionCoverageSummary {
    region_id: u32,
    sample_count: usize,
    covered_count: usize,
    empirical_coverage: f64,
}

/// Offline audit that compares aggregate coverage with coverage of points chosen by an optimizer.
/// This diagnostic does not refit, adjust, or certify the supplied conformal multiplier.
#[derive(Debug, Clone, PartialEq)]
struct PostSelectionCoverageAudit {
    nominal_coverage: f64,
    multiplier: f64,
    total_count: usize,
    covered_count: usize,
    overall_coverage: f64,
    selected_count: usize,
    selected_covered_count: usize,
    selected_coverage: f64,
    unselected_count: usize,
    unselected_covered_count: usize,
    unselected_coverage: f64,
    selected_minus_unselected_coverage: f64,
    worst_region_id: u32,
    worst_region_coverage: f64,
    regions: Vec<RegionCoverageSummary>,
}

/// Audit a fixed multiplier on labeled held-out residuals, with per-region minimum sample gates.
///
/// A selected/unselected gap or low regional coverage is evidence of a possible reliability
/// problem, not proof of conditional miscoverage in the population. Use a separate holdout;
/// never tune the multiplier on the same observations used for this audit.
fn audit_post_selection_coverage(
    samples: &[PostSelectionCoverageSample],
    multiplier: f64,
    nominal_coverage: f64,
    minimum_region_samples: usize,
) -> Result<PostSelectionCoverageAudit, ParetoMetricError> {
    if !multiplier.is_finite()
        || multiplier < 0.0
        || !nominal_coverage.is_finite()
        || nominal_coverage <= 0.0
        || nominal_coverage >= 1.0
        || minimum_region_samples == 0
    {
        return Err(ParetoMetricError::InvalidCoverageAuditParameter);
    }
    if samples.is_empty() {
        return Err(ParetoMetricError::EmptyCoverageAuditSet);
    }

    let mut selected_count = 0usize;
    let mut selected_covered_count = 0usize;
    let mut unselected_count = 0usize;
    let mut unselected_covered_count = 0usize;
    let mut total_covered_count = 0usize;
    let mut region_counts: std::collections::BTreeMap<u32, (usize, usize)> =
        std::collections::BTreeMap::new();

    for sample in samples {
        if !sample.standardized_residual_score.is_finite()
            || sample.standardized_residual_score < 0.0
        {
            return Err(ParetoMetricError::InvalidResidualScore);
        }
        let covered = sample.standardized_residual_score <= multiplier;
        let entry = region_counts.entry(sample.region_id).or_insert((0, 0));
        entry.0 += 1;
        if covered {
            entry.1 += 1;
            total_covered_count += 1;
        }

        if sample.optimizer_selected {
            selected_count += 1;
            if covered {
                selected_covered_count += 1;
            }
        } else {
            unselected_count += 1;
            if covered {
                unselected_covered_count += 1;
            }
        }
    }

    if selected_count == 0 {
        return Err(ParetoMetricError::MissingSelectedCoverage);
    }
    if unselected_count == 0 {
        return Err(ParetoMetricError::MissingUnselectedCoverage);
    }
    if region_counts
        .values()
        .any(|(count, _)| *count < minimum_region_samples)
    {
        return Err(ParetoMetricError::InsufficientCoverageGroupSample);
    }

    let mut regions = Vec::with_capacity(region_counts.len());
    let mut worst_region: Option<(u32, f64)> = None;
    for (region_id, (sample_count, covered_count)) in region_counts {
        let coverage = covered_count as f64 / sample_count as f64;
        if !coverage.is_finite() {
            return Err(ParetoMetricError::NonFiniteCalibrationMetric);
        }
        match worst_region {
            None => worst_region = Some((region_id, coverage)),
            Some((_, current_worst)) if coverage < current_worst => {
                worst_region = Some((region_id, coverage));
            }
            _ => {}
        }
        regions.push(RegionCoverageSummary {
            region_id,
            sample_count,
            covered_count,
            empirical_coverage: coverage,
        });
    }

    let total_count = samples.len();
    let overall_coverage = total_covered_count as f64 / total_count as f64;
    let selected_coverage = selected_covered_count as f64 / selected_count as f64;
    let unselected_coverage = unselected_covered_count as f64 / unselected_count as f64;
    let selected_minus_unselected_coverage = selected_coverage - unselected_coverage;
    let (worst_region_id, worst_region_coverage) =
        worst_region.ok_or(ParetoMetricError::EmptyCoverageAuditSet)?;
    if !overall_coverage.is_finite()
        || !selected_coverage.is_finite()
        || !unselected_coverage.is_finite()
        || !selected_minus_unselected_coverage.is_finite()
    {
        return Err(ParetoMetricError::NonFiniteCalibrationMetric);
    }

    Ok(PostSelectionCoverageAudit {
        nominal_coverage,
        multiplier,
        total_count,
        covered_count: total_covered_count,
        overall_coverage,
        selected_count,
        selected_covered_count,
        selected_coverage,
        unselected_count,
        unselected_covered_count,
        unselected_coverage,
        selected_minus_unselected_coverage,
        worst_region_id,
        worst_region_coverage,
        regions,
    })
}

/// One normalized-residual calibration observation with an importance weight
/// proportional to target/source covariate-density ratio.
#[derive(Debug, Clone, Copy, PartialEq)]
struct WeightedPredictionCalibrationSample {
    sample: PredictionCalibrationSample,
    importance_weight: f64,
}

/// Weighted conformal multiplier for one explicitly weighted test candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
struct WeightedSplitConformalMultiplier {
    alpha: f64,
    multiplier: f64,
    calibration_count: usize,
    weighted_effective_sample_size: f64,
    test_point_weight_share: f64,
    finite_quantile_mass_fraction: f64,
}

/// Fit a weighted split-conformal multiplier under a specified covariate-shift model.
///
/// test_point_importance_weight is the target/source density ratio for the new candidate.
/// The weighted conformal distribution places this test point's weight at positive infinity.
/// Thus finite calibration scores must accumulate weight at least
/// (1-alpha) * (sum calibration weights + test-point weight). If not, this function
/// refuses to emit a finite multiplier instead of dropping the test-point mass.
///
/// This only has a coverage interpretation if importance ratios and covariate-shift
/// assumptions are appropriate. It does not handle arbitrary conditional/label shift.
fn fit_weighted_split_conformal_multiplier(
    calibration: &[WeightedPredictionCalibrationSample],
    test_point_importance_weight: f64,
    alpha: f64,
) -> Result<WeightedSplitConformalMultiplier, ParetoMetricError> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(ParetoMetricError::InvalidMiscoverage);
    }
    if calibration.is_empty() {
        return Err(ParetoMetricError::EmptyCalibrationSet);
    }
    if !test_point_importance_weight.is_finite() || test_point_importance_weight <= 0.0 {
        return Err(ParetoMetricError::InvalidImportanceWeight);
    }

    let mut scored = Vec::with_capacity(calibration.len());
    let mut max_weight = test_point_importance_weight;
    for item in calibration {
        if !item.importance_weight.is_finite() || item.importance_weight < 0.0 {
            return Err(ParetoMetricError::InvalidImportanceWeight);
        }
        let score = standardized_absolute_residual(item.sample)?;
        max_weight = max_weight.max(item.importance_weight);
        scored.push((score, item.importance_weight));
    }
    if !max_weight.is_finite() || max_weight <= 0.0 {
        return Err(ParetoMetricError::InvalidImportanceWeight);
    }

    // Common scaling leaves weighted quantiles and ESS unchanged while reducing overflow.
    for (_, weight) in &mut scored {
        let normalized = *weight / max_weight;
        if *weight > 0.0 && normalized == 0.0 {
            return Err(ParetoMetricError::NonFiniteWeightedConformalMetric);
        }
        *weight = normalized;
    }
    let normalized_test_weight = test_point_importance_weight / max_weight;
    if !normalized_test_weight.is_finite() || normalized_test_weight <= 0.0 {
        return Err(ParetoMetricError::NonFiniteWeightedConformalMetric);
    }

    let calibration_weight_sum: f64 = scored.iter().map(|(_, weight)| *weight).sum();
    let squared_weight_sum: f64 = scored.iter().map(|(_, weight)| *weight * *weight).sum();
    if !calibration_weight_sum.is_finite() || !squared_weight_sum.is_finite() {
        return Err(ParetoMetricError::NonFiniteWeightedConformalMetric);
    }
    if calibration_weight_sum <= 0.0 || squared_weight_sum <= 0.0 {
        return Err(ParetoMetricError::ZeroTotalCalibrationWeight);
    }

    let total_weight = calibration_weight_sum + normalized_test_weight;
    let required_finite_mass = (1.0 - alpha) * total_weight;
    let weighted_effective_sample_size =
        calibration_weight_sum * calibration_weight_sum / squared_weight_sum;
    if !total_weight.is_finite()
        || !required_finite_mass.is_finite()
        || !weighted_effective_sample_size.is_finite()
        || weighted_effective_sample_size <= 0.0
    {
        return Err(ParetoMetricError::NonFiniteWeightedConformalMetric);
    }
    if required_finite_mass > calibration_weight_sum {
        return Err(ParetoMetricError::InsufficientWeightedCalibrationSupport);
    }

    scored.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut cumulative_weight = 0.0;
    let mut selected_multiplier = None;
    let mut selected_cumulative_weight = 0.0;
    for (score, weight) in scored {
        cumulative_weight += weight;
        if !cumulative_weight.is_finite() {
            return Err(ParetoMetricError::NonFiniteWeightedConformalMetric);
        }
        if cumulative_weight >= required_finite_mass {
            selected_multiplier = Some(score);
            selected_cumulative_weight = cumulative_weight;
            break;
        }
    }
    let multiplier =
        selected_multiplier.ok_or(ParetoMetricError::InsufficientWeightedCalibrationSupport)?;
    let finite_quantile_mass_fraction = selected_cumulative_weight / total_weight;
    let test_point_weight_share = normalized_test_weight / total_weight;
    if !finite_quantile_mass_fraction.is_finite()
        || !test_point_weight_share.is_finite()
    {
        return Err(ParetoMetricError::NonFiniteWeightedConformalMetric);
    }

    Ok(WeightedSplitConformalMultiplier {
        alpha,
        multiplier,
        calibration_count: calibration.len(),
        weighted_effective_sample_size,
        test_point_weight_share,
        finite_quantile_mass_fraction,
    })
}

/// Only surrogate outputs are accepted by the proposal policy. This type intentionally
/// has no evaluator truth, feasibility oracle, or hidden-optimum field.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PredictedObjectiveCandidate {
    candidate_id: u64,
    predicted_mean: [f64; 2],
    predictive_std_dev: [f64; 2],
}

/// Auditable receipt for a largest-gap epsilon-constrained proposal.
#[derive(Debug, Clone, Copy, PartialEq)]
struct EpsilonGapProposal {
    selected_candidate_id: u64,
    lower_gap_endpoint_trial_id: u64,
    upper_gap_endpoint_trial_id: u64,
    epsilon: f64,
    normalized_gap: f64,
    confidence_multipliers: [f64; 2],
    objective_0_lower_confidence_bound: f64,
    objective_1_upper_confidence_bound: f64,
}

/// Propose one point using a simple coverage-seeking epsilon-constraint baseline.
///
/// The largest adjacent gap is measured after dividing each objective by a frozen
/// positive scale. The midpoint of the gap's second-objective coordinates becomes
/// epsilon. Candidates are filtered by mu_1 + beta*sigma_1 <= epsilon, then ranked
/// by mu_0 - beta*sigma_0, with stable numeric and ID tie-breaks.
///
/// This is a transparent proposal policy, not STAGE-BO and not a BO acquisition function.
/// The caller is responsible for supplying predictions from an independently defined
/// surrogate; this function never evaluates candidates or receives hidden truth.
fn propose_largest_gap_epsilon(
    front: &[ParetoPoint],
    candidates: &[PredictedObjectiveCandidate],
    objective_scales: [f64; 2],
    confidence_multipliers: [f64; 2],
) -> Result<EpsilonGapProposal, ParetoMetricError> {
    if front.len() < 2 {
        return Err(ParetoMetricError::InsufficientFrontForGap);
    }
    if candidates.is_empty() {
        return Err(ParetoMetricError::EmptyPredictedCandidateSet);
    }
    if objective_scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(ParetoMetricError::InvalidObjectiveScale);
    }
    if confidence_multipliers
        .iter()
        .any(|multiplier| !multiplier.is_finite() || *multiplier < 0.0)
    {
        return Err(ParetoMetricError::InvalidConfidenceMultiplier);
    }

    let mut ordered_front = front.to_vec();
    if ordered_front
        .iter()
        .any(|point| point.objectives.iter().any(|value| !value.is_finite()))
    {
        return Err(ParetoMetricError::NonFiniteObjective);
    }
    ordered_front.sort_by(|left, right| {
        left.objectives[0]
            .total_cmp(&right.objectives[0])
            .then_with(|| left.objectives[1].total_cmp(&right.objectives[1]))
            .then_with(|| left.trial_id.cmp(&right.trial_id))
    });
    for pair in ordered_front.windows(2) {
        if pair[0].objectives == pair[1].objectives {
            return Err(ParetoMetricError::InsufficientFrontForGap);
        }
    }
    let mut seen_front_ids = std::collections::HashSet::with_capacity(ordered_front.len());
    for point in &ordered_front {
        if !seen_front_ids.insert(point.trial_id) {
            return Err(ParetoMetricError::DuplicateTrialId);
        }
    }
    for (index, left) in ordered_front.iter().enumerate() {
        if ordered_front
            .iter()
            .enumerate()
            .any(|(other_index, right)| index != other_index && dominates(left.objectives, right.objectives))
        {
            return Err(ParetoMetricError::InvalidParetoFront);
        }
    }

    let mut best_gap: Option<(usize, f64)> = None;
    for index in 0..ordered_front.len() - 1 {
        let left = ordered_front[index].objectives;
        let right = ordered_front[index + 1].objectives;
        let delta_0 = (right[0] - left[0]) / objective_scales[0];
        let delta_1 = (right[1] - left[1]) / objective_scales[1];
        if !delta_0.is_finite() || !delta_1.is_finite() {
            return Err(ParetoMetricError::NonFiniteMetricResult);
        }
        let gap = delta_0.hypot(delta_1);
        if !gap.is_finite() {
            return Err(ParetoMetricError::NonFiniteMetricResult);
        }
        match best_gap {
            None => best_gap = Some((index, gap)),
            Some((_, best_value)) => {
                if gap > best_value {
                    best_gap = Some((index, gap));
                }
                // Ordered front + first-match policy makes exact ties deterministic.
            }
        }
    }
    let (gap_index, normalized_gap) = best_gap.ok_or(ParetoMetricError::InsufficientFrontForGap)?;
    if normalized_gap <= 0.0 {
        return Err(ParetoMetricError::InsufficientFrontForGap);
    }
    let left = ordered_front[gap_index];
    let right = ordered_front[gap_index + 1];
    let epsilon = left.objectives[1] / 2.0 + right.objectives[1] / 2.0;
    if !epsilon.is_finite() {
        return Err(ParetoMetricError::NonFiniteMetricResult);
    }

    let mut seen_ids = std::collections::HashSet::with_capacity(candidates.len());
    let mut best: Option<(PredictedObjectiveCandidate, f64, f64)> = None;
    for candidate in candidates {
        if !seen_ids.insert(candidate.candidate_id) {
            return Err(ParetoMetricError::DuplicateCandidateId);
        }
        if candidate.predicted_mean.iter().any(|value| !value.is_finite())
            || candidate
                .predictive_std_dev
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(ParetoMetricError::InvalidPredictionUncertainty);
        }
        let uncertainty_0 = confidence_multipliers[0] * candidate.predictive_std_dev[0];
        let uncertainty_1 = confidence_multipliers[1] * candidate.predictive_std_dev[1];
        if !uncertainty_0.is_finite() || !uncertainty_1.is_finite() {
            return Err(ParetoMetricError::InvalidPredictionUncertainty);
        }
        let lcb_0 = candidate.predicted_mean[0] - uncertainty_0;
        let ucb_1 = candidate.predicted_mean[1] + uncertainty_1;
        if !lcb_0.is_finite() || !ucb_1.is_finite() {
            return Err(ParetoMetricError::InvalidPredictionUncertainty);
        }
        if ucb_1 > epsilon {
            continue;
        }

        let should_replace = match best {
            None => true,
            Some((current, current_lcb, current_ucb)) => {
                lcb_0 < current_lcb
                    || (lcb_0 == current_lcb
                        && (ucb_1 < current_ucb
                            || (ucb_1 == current_ucb
                                && candidate.candidate_id < current.candidate_id)))
            }
        };
        if should_replace {
            best = Some((*candidate, lcb_0, ucb_1));
        }
    }

    let (selected, lcb_0, ucb_1) =
        best.ok_or(ParetoMetricError::NoEpsilonFeasibleCandidate)?;
    Ok(EpsilonGapProposal {
        selected_candidate_id: selected.candidate_id,
        lower_gap_endpoint_trial_id: left.trial_id,
        upper_gap_endpoint_trial_id: right.trial_id,
        epsilon,
        normalized_gap,
        confidence_multipliers,
        objective_0_lower_confidence_bound: lcb_0,
        objective_1_upper_confidence_bound: ucb_1,
    })
}

/// Canonical two-objective ZDT benchmark families used for known-front validation.
/// All objectives are minimized. These are standard synthetic optimization fixtures,
/// not casting physics or material-process models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ZdtProblem {
    Zdt1,
    Zdt2,
    Zdt3,
    Zdt4,
}

/// ZDT1/2/3 use 30 variables in [0, 1]; ZDT4 uses 10 variables, x1 in [0, 1],
/// and remaining variables in [-10, 10], matching the standard benchmark dimensions.
fn canonical_zdt_dimension(problem: ZdtProblem) -> usize {
    match problem {
        ZdtProblem::Zdt1 | ZdtProblem::Zdt2 | ZdtProblem::Zdt3 => 30,
        ZdtProblem::Zdt4 => 10,
    }
}

/// Construct a Pareto-set decision with x1 = f1 and all tail variables set to zero.
fn zdt_pareto_decision(problem: ZdtProblem, f1: f64) -> Vec<f64> {
    let mut decision = vec![0.0; canonical_zdt_dimension(problem)];
    decision[0] = f1;
    decision
}

/// Evaluate a ZDT decision vector using canonical benchmark dimensions and domains.
fn evaluate_zdt(problem: ZdtProblem, decision: &[f64]) -> Result<[f64; 2], ParetoMetricError> {
    if decision.len() != canonical_zdt_dimension(problem) {
        return Err(ParetoMetricError::InvalidDecisionDimension);
    }
    if decision.iter().any(|value| !value.is_finite()) {
        return Err(ParetoMetricError::NonFiniteDecisionVariable);
    }
    let first_is_valid = (0.0..=1.0).contains(&decision[0]);
    let tail_is_valid = match problem {
        ZdtProblem::Zdt4 => decision[1..].iter().all(|value| (-10.0..=10.0).contains(value)),
        ZdtProblem::Zdt1 | ZdtProblem::Zdt2 | ZdtProblem::Zdt3 => {
            decision[1..].iter().all(|value| (0.0..=1.0).contains(value))
        }
    };
    if !first_is_valid || !tail_is_valid {
        return Err(ParetoMetricError::DecisionOutOfDomain);
    }

    let f1 = decision[0];
    let g = match problem {
        ZdtProblem::Zdt4 => {
            1.0 + 10.0 * (decision.len() - 1) as f64
                + decision[1..]
                    .iter()
                    .map(|value| value * value - 10.0 * (4.0 * std::f64::consts::PI * value).cos())
                    .sum::<f64>()
        }
        ZdtProblem::Zdt1 | ZdtProblem::Zdt2 | ZdtProblem::Zdt3 => {
            1.0 + 9.0 * decision[1..].iter().sum::<f64>() / (decision.len() - 1) as f64
        }
    };
    if !g.is_finite() || g <= 0.0 {
        return Err(ParetoMetricError::NonFiniteMetricResult);
    }

    let ratio = f1 / g;
    let h = match problem {
        ZdtProblem::Zdt1 | ZdtProblem::Zdt4 => 1.0 - ratio.sqrt(),
        ZdtProblem::Zdt2 => 1.0 - ratio * ratio,
        ZdtProblem::Zdt3 => {
            1.0 - ratio.sqrt() - ratio * (10.0 * std::f64::consts::PI * f1).sin()
        }
    };
    let f2 = g * h;
    if !f1.is_finite() || !f2.is_finite() {
        return Err(ParetoMetricError::NonFiniteMetricResult);
    }
    Ok([f1, f2])
}

/// Generate a deterministic sampled approximation to the analytical Pareto front.
/// points_per_segment includes both endpoints of each sampled interval and must be >= 2.
/// ZDT3's five separated intervals are sampled independently; gaps are never bridged.
/// Its later four intervals have open lower boundaries; approximate published bounds are
/// offset inward by 1e-8 to avoid a rounded, dominated boundary point. Sampling is uniform
/// in f1 parameter, not Euclidean arc length. Freeze this convention for epsilon/IGD+.
fn zdt_reference_front(
    problem: ZdtProblem,
    points_per_segment: usize,
) -> Result<Vec<[f64; 2]>, ParetoMetricError> {
    if points_per_segment < 2 {
        return Err(ParetoMetricError::InvalidReferenceFrontResolution);
    }

    let intervals: &[(f64, f64)] = match problem {
        ZdtProblem::Zdt1 | ZdtProblem::Zdt2 | ZdtProblem::Zdt4 => &[(0.0, 1.0)],
        ZdtProblem::Zdt3 => &[
            (0.0, 0.0830015349),
            (0.1822287280, 0.2577623634),
            (0.4093136748, 0.4538821041),
            (0.6183967944, 0.6525117038),
            (0.8233317983, 0.8518328654),
        ],
    };

    let mut front = Vec::with_capacity(intervals.len() * points_per_segment);
    for (segment_index, &(start, end)) in intervals.iter().enumerate() {
        let sampled_start = if problem == ZdtProblem::Zdt3 && segment_index > 0 {
            start + 1e-8
        } else {
            start
        };
        for sample_index in 0..points_per_segment {
            let fraction = sample_index as f64 / (points_per_segment - 1) as f64;
            let f1 = sampled_start + (end - sampled_start) * fraction;
            let f2 = match problem {
                ZdtProblem::Zdt1 | ZdtProblem::Zdt4 => 1.0 - f1.sqrt(),
                ZdtProblem::Zdt2 => 1.0 - f1 * f1,
                ZdtProblem::Zdt3 => {
                    1.0 - f1.sqrt() - f1 * (10.0 * std::f64::consts::PI * f1).sin()
                }
            };
            if !f1.is_finite() || !f2.is_finite() {
                return Err(ParetoMetricError::NonFiniteMetricResult);
            }
            front.push([f1, f2]);
        }
    }
    Ok(front)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Selection-shift Monte Carlo stress benchmark, kept separate for readability.
    mod selection_shift_monte_carlo;

    fn feasible(trial_id: u64, first: f64, second: f64, cost_units: u64) -> ParetoTrial {
        ParetoTrial {
            trial_id,
            status: ParetoTrialStatus::Feasible,
            objectives: Some([first, second]),
            cost_units,
        }
    }

    fn failed(trial_id: u64, cost_units: u64) -> ParetoTrial {
        ParetoTrial {
            trial_id,
            status: ParetoTrialStatus::Failed,
            objectives: None,
            cost_units,
        }
    }

    fn infeasible(trial_id: u64, cost_units: u64) -> ParetoTrial {
        ParetoTrial {
            trial_id,
            status: ParetoTrialStatus::Infeasible,
            objectives: None,
            cost_units,
        }
    }

    fn sample_archive(order: &[(u64, f64, f64)]) -> ParetoArchive {
        let mut archive = ParetoArchive::new([10.0, 10.0]).unwrap();
        for &(trial_id, first, second) in order {
            archive
                .observe(feasible(trial_id, first, second, 1))
                .unwrap();
        }
        archive
    }

    #[test]
    fn empty_archive_has_zero_hypervolume() {
        let archive = ParetoArchive::new([10.0, 10.0]).unwrap();
        assert_eq!(archive.hypervolume(), 0.0);
        assert!(archive.front().is_empty());
        assert!(archive.trace().is_empty());
    }

    #[test]
    fn known_three_point_front_has_exact_hypervolume_46() {
        // Minimize both objectives; reference point is fixed at (10, 10).
        // x-slices: (4 - 2)*(10 - 8) + (7 - 4)*(10 - 4) + (10 - 7)*(10 - 2) = 46.
        let archive = sample_archive(&[(1, 2.0, 8.0), (2, 4.0, 4.0), (3, 7.0, 2.0)]);
        assert_abs_close(archive.hypervolume(), 46.0, 1e-12);
        assert_eq!(archive.front().len(), 3);
    }

    #[test]
    fn dominated_points_are_removed_without_hiding_the_attempt() {
        let mut archive = sample_archive(&[(1, 2.0, 8.0), (2, 4.0, 4.0), (3, 7.0, 2.0)]);
        let previous_hypervolume = archive.hypervolume();
        archive.observe(feasible(4, 5.0, 8.0, 3)).unwrap();
        assert_eq!(archive.front().len(), 3);
        assert_eq!(archive.hypervolume(), previous_hypervolume);
        assert_eq!(archive.attempt_count(), 4);
        assert_eq!(archive.total_cost_units(), 6);
    }

    #[test]
    fn front_and_hypervolume_are_invariant_to_trial_order() {
        let points = [(1, 2.0, 8.0), (2, 4.0, 4.0), (3, 7.0, 2.0), (4, 5.0, 8.0)];
        let forward = sample_archive(&points);
        let reverse_order = [(4, 5.0, 8.0), (3, 7.0, 2.0), (2, 4.0, 4.0), (1, 2.0, 8.0)];
        let reverse = sample_archive(&reverse_order);
        assert_eq!(forward.front(), reverse.front());
        assert_abs_close(forward.hypervolume(), reverse.hypervolume(), 1e-12);
    }

    #[test]
    fn failed_and_infeasible_trials_cost_resources_but_do_not_enter_front() {
        let mut archive = ParetoArchive::new([10.0, 10.0]).unwrap();
        archive.observe(failed(1, 7)).unwrap();
        archive.observe(infeasible(2, 3)).unwrap();
        archive.observe(feasible(3, 2.0, 8.0, 10)).unwrap();

        assert_eq!(archive.status_counts(), (1, 1, 1));
        assert_eq!(archive.attempt_count(), 3);
        assert_eq!(archive.total_cost_units(), 20);
        assert_eq!(archive.front().len(), 1);
        assert_abs_close(archive.hypervolume(), 16.0, 1e-12);
    }

    #[test]
    fn cost_to_hypervolume_target_includes_prior_failed_and_infeasible_attempts() {
        let mut archive = ParetoArchive::new([10.0, 10.0]).unwrap();
        archive.observe(failed(1, 7)).unwrap();
        archive.observe(feasible(2, 2.0, 8.0, 10)).unwrap();
        archive.observe(infeasible(3, 4)).unwrap();
        archive.observe(feasible(4, 4.0, 4.0, 10)).unwrap();

        // HV becomes 16 after trial 2 and 40 after trial 4; attempted costs are cumulative.
        assert_eq!(archive.cost_to_hypervolume_target(18.0).unwrap(), Some(31));
        assert_eq!(archive.cost_to_hypervolume_target(45.0).unwrap(), None);
        assert_eq!(archive.trace()[2].hypervolume, 16.0);
        assert_eq!(archive.trace()[2].cumulative_cost_units, 21);
    }

    #[test]
    fn invalid_objectives_are_rejected_atomically() {
        let mut archive = ParetoArchive::new([10.0, 10.0]).unwrap();
        archive.observe(feasible(1, 2.0, 8.0, 5)).unwrap();
        let before = archive.clone();

        assert_eq!(
            archive.observe(feasible(2, f64::NAN, 4.0, 8)),
            Err(ParetoMetricError::NonFiniteObjective)
        );
        assert_eq!(
            archive.observe(feasible(3, 10.0, 4.0, 8)),
            Err(ParetoMetricError::OutsideReferenceBox)
        );
        assert_eq!(archive, before);
    }

    #[test]
    fn duplicate_ids_inconsistent_payloads_and_zero_cost_fail_atomically() {
        let mut archive = ParetoArchive::new([10.0, 10.0]).unwrap();
        archive.observe(feasible(1, 2.0, 8.0, 5)).unwrap();
        let before = archive.clone();

        assert_eq!(
            archive.observe(feasible(1, 4.0, 4.0, 3)),
            Err(ParetoMetricError::DuplicateTrialId)
        );
        assert_eq!(
            archive.observe(ParetoTrial {
                trial_id: 2,
                status: ParetoTrialStatus::Failed,
                objectives: Some([1.0, 1.0]),
                cost_units: 3,
            }),
            Err(ParetoMetricError::InconsistentOutcome)
        );
        assert_eq!(
            archive.observe(feasible(2, 4.0, 4.0, 0)),
            Err(ParetoMetricError::InvalidCost)
        );
        assert_eq!(archive, before);
    }

    #[test]
    fn invalid_reference_points_and_targets_fail_closed() {
        assert_eq!(
            ParetoArchive::new([10.0, f64::INFINITY]),
            Err(ParetoMetricError::InvalidReferencePoint)
        );
        let archive = ParetoArchive::new([10.0, 10.0]).unwrap();
        assert_eq!(
            archive.cost_to_hypervolume_target(0.0),
            Err(ParetoMetricError::InvalidHypervolumeTarget)
        );
        assert_eq!(
            archive.cost_to_hypervolume_target(f64::NAN),
            Err(ParetoMetricError::InvalidHypervolumeTarget)
        );
    }

    #[test]
    fn objective_duplicates_keep_the_lowest_id_representative() {
        let archive = sample_archive(&[(9, 2.0, 8.0), (4, 2.0, 8.0), (7, 4.0, 4.0)]);
        assert_eq!(archive.front().len(), 2);
        assert_eq!(archive.front()[0].trial_id, 4);
        assert_abs_close(archive.hypervolume(), 40.0, 1e-12);
    }

    #[test]
    fn cost_overflow_is_rejected_without_partial_mutation() {
        let mut archive = ParetoArchive::new([10.0, 10.0]).unwrap();
        archive.observe(feasible(1, 2.0, 8.0, u64::MAX)).unwrap();
        let before = archive.clone();
        assert_eq!(
            archive.observe(feasible(2, 4.0, 4.0, 1)),
            Err(ParetoMetricError::CostOverflow)
        );
        assert_eq!(archive, before);
    }


    #[test]
    fn additive_epsilon_indicator_matches_hand_calculation() {
        let approximation = [[2.0, 2.0], [4.0, 0.0]];
        let reference = [[1.0, 3.0], [3.0, 1.0]];
        assert_abs_close(
            additive_epsilon_indicator(&approximation, &reference).unwrap(),
            1.0,
            1e-12,
        );
    }

    #[test]
    fn additive_epsilon_can_be_negative_when_approximation_dominates_reference() {
        assert_abs_close(
            additive_epsilon_indicator(&[[1.0, 1.0]], &[[2.0, 2.0]]).unwrap(),
            -1.0,
            1e-12,
        );
    }

    #[test]
    fn igd_plus_matches_hand_calculation_and_rewards_dominance() {
        let approximation = [[1.0, 3.0], [3.0, 1.0]];
        let reference = [[1.0, 2.0], [2.0, 1.0]];
        assert_abs_close(
            inverted_generational_distance_plus(&approximation, &reference).unwrap(),
            1.0,
            1e-12,
        );
        assert_eq!(
            inverted_generational_distance_plus(&[[1.0, 1.0]], &[[2.0, 2.0]]).unwrap(),
            0.0
        );
    }

    #[test]
    fn spacing_matches_sample_standard_deviation_of_nearest_distances() {
        let front = [[0.0, 0.0], [3.0, 0.0], [10.0, 0.0]];
        assert_abs_close(
            nearest_neighbour_spacing(&front).unwrap(),
            (16.0_f64 / 3.0).sqrt(),
            1e-12,
        );
    }

    #[test]
    fn front_metrics_are_invariant_to_input_order() {
        let approximation = [[2.0, 2.0], [4.0, 0.0], [1.5, 2.5]];
        let reversed_approximation = [[1.5, 2.5], [4.0, 0.0], [2.0, 2.0]];
        let reference = [[1.0, 3.0], [3.0, 1.0], [2.5, 2.0]];
        let reversed_reference = [[2.5, 2.0], [3.0, 1.0], [1.0, 3.0]];
        assert_abs_close(
            additive_epsilon_indicator(&approximation, &reference).unwrap(),
            additive_epsilon_indicator(&reversed_approximation, &reversed_reference).unwrap(),
            1e-12,
        );
        assert_abs_close(
            inverted_generational_distance_plus(&approximation, &reference).unwrap(),
            inverted_generational_distance_plus(
                &reversed_approximation,
                &reversed_reference,
            )
            .unwrap(),
            1e-12,
        );
        assert_abs_close(
            nearest_neighbour_spacing(&approximation).unwrap(),
            nearest_neighbour_spacing(&reversed_approximation).unwrap(),
            1e-12,
        );
    }

    #[test]
    fn front_metrics_reject_empty_and_non_finite_inputs() {
        assert_eq!(
            additive_epsilon_indicator(&[], &[[1.0, 1.0]]),
            Err(ParetoMetricError::EmptyApproximationFront)
        );
        assert_eq!(
            inverted_generational_distance_plus(&[[1.0, 1.0]], &[]),
            Err(ParetoMetricError::EmptyReferenceFront)
        );
        assert_eq!(
            nearest_neighbour_spacing(&[[1.0, 1.0]]),
            Err(ParetoMetricError::TooFewPointsForSpacing)
        );
        assert_eq!(
            additive_epsilon_indicator(&[[f64::NAN, 0.0]], &[[1.0, 1.0]]),
            Err(ParetoMetricError::NonFiniteObjective)
        );
        assert_eq!(
            inverted_generational_distance_plus(&[[f64::MAX, 0.0]], &[[-f64::MAX, 0.0]]),
            Err(ParetoMetricError::NonFiniteMetricResult)
        );
    }


    #[test]
    fn zdt1_zdt2_and_zdt3_match_known_objective_values() {
        let zdt1_decision = zdt_pareto_decision(ZdtProblem::Zdt1, 0.5);
        let zdt1 = evaluate_zdt(ZdtProblem::Zdt1, &zdt1_decision).unwrap();
        assert_abs_close(zdt1[0], 0.5, 1e-12);
        assert_abs_close(zdt1[1], 1.0 - 0.5_f64.sqrt(), 1e-12);

        let zdt2_decision = zdt_pareto_decision(ZdtProblem::Zdt2, 0.5);
        let zdt2 = evaluate_zdt(ZdtProblem::Zdt2, &zdt2_decision).unwrap();
        assert_abs_close(zdt2[0], 0.5, 1e-12);
        assert_abs_close(zdt2[1], 0.75, 1e-12);

        let zdt3_decision = zdt_pareto_decision(ZdtProblem::Zdt3, 0.5);
        let zdt3 = evaluate_zdt(ZdtProblem::Zdt3, &zdt3_decision).unwrap();
        assert_abs_close(zdt3[0], 0.5, 1e-12);
        assert_abs_close(zdt3[1], 1.0 - 0.5_f64.sqrt(), 1e-12);
    }

    #[test]
    fn zdt4_matches_known_front_and_exposes_multimodal_tail_penalty() {
        let optimum_decision = zdt_pareto_decision(ZdtProblem::Zdt4, 0.5);
        let optimum = evaluate_zdt(ZdtProblem::Zdt4, &optimum_decision).unwrap();
        assert_abs_close(optimum[1], 1.0 - 0.5_f64.sqrt(), 1e-12);

        // A tail coordinate away from its global optimum increases g on the canonical domain.
        let mut non_optimal_decision = zdt_pareto_decision(ZdtProblem::Zdt4, 0.5);
        non_optimal_decision[1] = 1.0;
        let non_optimal = evaluate_zdt(ZdtProblem::Zdt4, &non_optimal_decision).unwrap();
        let mut in_domain_multimodal = zdt_pareto_decision(ZdtProblem::Zdt4, 0.5);
        in_domain_multimodal[1] = 6.0;
        assert!(evaluate_zdt(ZdtProblem::Zdt4, &in_domain_multimodal).is_ok());
        assert_abs_close(non_optimal[1], 1.0, 1e-12);
        assert!(non_optimal[1] > optimum[1]);
    }

    #[test]
    fn zdt_reference_fronts_are_deterministic_and_correctly_sized() {
        for problem in [ZdtProblem::Zdt1, ZdtProblem::Zdt2, ZdtProblem::Zdt4] {
            let first = zdt_reference_front(problem, 101).unwrap();
            let second = zdt_reference_front(problem, 101).unwrap();
            assert_eq!(first, second);
            assert_eq!(first.len(), 101);
        }

        let zdt3 = zdt_reference_front(ZdtProblem::Zdt3, 21).unwrap();
        assert_eq!(zdt3.len(), 5 * 21);
        // Segment boundaries are explicit; no sampled interval bridges a disconnected gap.
        assert!(zdt3[20][0] < zdt3[21][0]);
        assert!(zdt3[41][0] < zdt3[42][0]);
        assert!(zdt3[62][0] < zdt3[63][0]);
        assert!(zdt3[83][0] < zdt3[84][0]);
    }

    #[test]
    fn each_reference_front_point_is_attainable_by_a_pareto_set_decision() {
        for problem in [ZdtProblem::Zdt1, ZdtProblem::Zdt2, ZdtProblem::Zdt3, ZdtProblem::Zdt4] {
            let front = zdt_reference_front(problem, 17).unwrap();
            for expected in &front {
                let decision = zdt_pareto_decision(problem, expected[0]);
                let actual = evaluate_zdt(problem, &decision).unwrap();
                assert_abs_close(actual[0], expected[0], 1e-10);
                assert_abs_close(actual[1], expected[1], 1e-8);
            }
        }
    }

    #[test]
    fn sampled_zdt_reference_fronts_are_nondominated_and_metric_known_answers() {
        for problem in [ZdtProblem::Zdt1, ZdtProblem::Zdt2, ZdtProblem::Zdt3, ZdtProblem::Zdt4] {
            let front = zdt_reference_front(problem, 9).unwrap();
            for (i, left) in front.iter().enumerate() {
                for (j, right) in front.iter().enumerate() {
                    if i != j {
                        assert!(!dominates(*left, *right), "point {i} dominates point {j} for {problem:?}");
                    }
                }
            }
            assert_abs_close(additive_epsilon_indicator(&front, &front).unwrap(), 0.0, 1e-12);
            assert_abs_close(inverted_generational_distance_plus(&front, &front).unwrap(), 0.0, 1e-12);
        }
    }

    #[test]
    fn zdt_evaluator_rejects_malformed_non_finite_and_out_of_domain_decisions() {
        assert_eq!(
            evaluate_zdt(ZdtProblem::Zdt1, &[0.5]),
            Err(ParetoMetricError::InvalidDecisionDimension)
        );
        let mut non_finite = zdt_pareto_decision(ZdtProblem::Zdt1, 0.5);
        non_finite[0] = f64::NAN;
        assert_eq!(
            evaluate_zdt(ZdtProblem::Zdt1, &non_finite),
            Err(ParetoMetricError::NonFiniteDecisionVariable)
        );
        let mut invalid_zdt1 = zdt_pareto_decision(ZdtProblem::Zdt1, 0.5);
        invalid_zdt1[1] = -0.01;
        assert_eq!(
            evaluate_zdt(ZdtProblem::Zdt1, &invalid_zdt1),
            Err(ParetoMetricError::DecisionOutOfDomain)
        );
        let mut invalid_zdt4 = zdt_pareto_decision(ZdtProblem::Zdt4, 0.5);
        invalid_zdt4[1] = 10.01;
        assert_eq!(
            evaluate_zdt(ZdtProblem::Zdt4, &invalid_zdt4),
            Err(ParetoMetricError::DecisionOutOfDomain)
        );
        assert_eq!(
            zdt_reference_front(ZdtProblem::Zdt3, 1),
            Err(ParetoMetricError::InvalidReferenceFrontResolution)
        );
    }


    #[test]
    fn zdt_reference_fixtures_match_known_archive_hypervolume() {
        let zdt1_front = zdt_reference_front(ZdtProblem::Zdt1, 3).unwrap();
        let mut zdt1_archive = ParetoArchive::new([2.0, 2.0]).unwrap();
        for (index, point) in zdt1_front.iter().enumerate() {
            zdt1_archive
                .observe(feasible(index as u64 + 1, point[0], point[1], 1))
                .unwrap();
        }
        let expected_zdt1_hv = 2.5 + 0.5 * (1.0 + 0.5_f64.sqrt());
        assert_abs_close(zdt1_archive.hypervolume(), expected_zdt1_hv, 1e-12);

        let zdt2_front = zdt_reference_front(ZdtProblem::Zdt2, 3).unwrap();
        let mut zdt2_archive = ParetoArchive::new([2.0, 2.0]).unwrap();
        for (index, point) in zdt2_front.iter().enumerate() {
            zdt2_archive
                .observe(feasible(index as u64 + 1, point[0], point[1], 1))
                .unwrap();
        }
        assert_abs_close(zdt2_archive.hypervolume(), 3.125, 1e-12);
    }


    fn gap_front() -> Vec<ParetoPoint> {
        vec![
            ParetoPoint { trial_id: 1, objectives: [1.0, 9.0] },
            ParetoPoint { trial_id: 2, objectives: [2.0, 7.0] },
            ParetoPoint { trial_id: 3, objectives: [4.0, 6.0] },
            ParetoPoint { trial_id: 4, objectives: [5.0, 1.0] },
        ]
    }

    fn gap_candidates() -> Vec<PredictedObjectiveCandidate> {
        vec![
            PredictedObjectiveCandidate {
                candidate_id: 10,
                predicted_mean: [2.0, 3.3],
                predictive_std_dev: [0.1, 0.1],
            },
            PredictedObjectiveCandidate {
                candidate_id: 11,
                predicted_mean: [1.8, 3.45],
                predictive_std_dev: [0.5, 0.1],
            },
            PredictedObjectiveCandidate {
                candidate_id: 12,
                predicted_mean: [1.6, 4.0],
                predictive_std_dev: [0.5, 0.5],
            },
        ]
    }

    #[test]
    fn epsilon_gap_policy_targets_largest_normalized_gap_and_respects_ucb() {
        let proposal = propose_largest_gap_epsilon(&gap_front(), &gap_candidates(), [1.0, 2.0], [1.0, 2.0]).unwrap();
        assert_eq!(proposal.lower_gap_endpoint_trial_id, 3);
        assert_eq!(proposal.upper_gap_endpoint_trial_id, 4);
        assert_abs_close(proposal.normalized_gap, 2.692582403567252, 1e-12);
        assert_abs_close(proposal.epsilon, 3.5, 1e-12);
        assert_eq!(proposal.selected_candidate_id, 10);
        assert!(proposal.objective_1_upper_confidence_bound <= proposal.epsilon);
        assert_abs_close(proposal.objective_0_lower_confidence_bound, 1.9, 1e-12);
        assert_abs_close(proposal.objective_1_upper_confidence_bound, 3.5, 1e-12);
    }

    #[test]
    fn epsilon_gap_policy_is_invariant_to_front_and_candidate_order() {
        let mut reversed_front = gap_front();
        reversed_front.reverse();
        let mut reversed_candidates = gap_candidates();
        reversed_candidates.reverse();
        let first = propose_largest_gap_epsilon(&gap_front(), &gap_candidates(), [1.0, 2.0], [1.0, 1.0]).unwrap();
        let reversed = propose_largest_gap_epsilon(&reversed_front, &reversed_candidates, [1.0, 2.0], [1.0, 1.0]).unwrap();
        assert_eq!(first, reversed);
    }

    #[test]
    fn epsilon_gap_policy_uses_candidate_id_as_final_tie_breaker() {
        let candidates = vec![
            PredictedObjectiveCandidate {
                candidate_id: 20,
                predicted_mean: [1.0, 3.0],
                predictive_std_dev: [0.0, 0.0],
            },
            PredictedObjectiveCandidate {
                candidate_id: 10,
                predicted_mean: [1.0, 3.0],
                predictive_std_dev: [0.0, 0.0],
            },
        ];
        let proposal = propose_largest_gap_epsilon(&gap_front(), &candidates, [1.0, 2.0], [0.0, 0.0]).unwrap();
        assert_eq!(proposal.selected_candidate_id, 10);
    }

    #[test]
    fn epsilon_gap_policy_fails_closed_on_empty_and_invalid_inputs() {
        assert_eq!(
            propose_largest_gap_epsilon(&gap_front()[..1], &gap_candidates(), [1.0, 2.0], [1.0, 1.0]),
            Err(ParetoMetricError::InsufficientFrontForGap)
        );
        assert_eq!(
            propose_largest_gap_epsilon(&gap_front(), &[], [1.0, 2.0], [1.0, 1.0]),
            Err(ParetoMetricError::EmptyPredictedCandidateSet)
        );
        assert_eq!(
            propose_largest_gap_epsilon(&gap_front(), &gap_candidates(), [0.0, 2.0], 1.0),
            Err(ParetoMetricError::InvalidObjectiveScale)
        );
        assert_eq!(
            propose_largest_gap_epsilon(&gap_front(), &gap_candidates(), [1.0, 2.0], [f64::NAN, 1.0]),
            Err(ParetoMetricError::InvalidConfidenceMultiplier)
        );
        let bad_uncertainty = [PredictedObjectiveCandidate {
            candidate_id: 1,
            predicted_mean: [1.0, 1.0],
            predictive_std_dev: [-0.1, 0.1],
        }];
        assert_eq!(
            propose_largest_gap_epsilon(&gap_front(), &bad_uncertainty, [1.0, 2.0], [1.0, 1.0]),
            Err(ParetoMetricError::InvalidPredictionUncertainty)
        );
    }

    #[test]
    fn epsilon_gap_policy_rejects_duplicate_ids_and_no_feasible_candidate() {
        let duplicate_ids = [
            PredictedObjectiveCandidate {
                candidate_id: 1,
                predicted_mean: [1.0, 2.0],
                predictive_std_dev: [0.0, 0.0],
            },
            PredictedObjectiveCandidate {
                candidate_id: 1,
                predicted_mean: [2.0, 2.0],
                predictive_std_dev: [0.0, 0.0],
            },
        ];
        assert_eq!(
            propose_largest_gap_epsilon(&gap_front(), &duplicate_ids, [1.0, 2.0], [1.0, 1.0]),
            Err(ParetoMetricError::DuplicateCandidateId)
        );
        let over_epsilon = [PredictedObjectiveCandidate {
            candidate_id: 2,
            predicted_mean: [0.5, 4.0],
            predictive_std_dev: [0.1, 0.1],
        }];
        assert_eq!(
            propose_largest_gap_epsilon(&gap_front(), &over_epsilon, [1.0, 2.0], [1.0, 1.0]),
            Err(ParetoMetricError::NoEpsilonFeasibleCandidate)
        );
    }


    #[test]
    fn epsilon_gap_policy_rejects_dominated_or_duplicate_front_trials() {
        let dominated = [
            ParetoPoint { trial_id: 1, objectives: [1.0, 1.0] },
            ParetoPoint { trial_id: 2, objectives: [2.0, 2.0] },
        ];
        assert_eq!(
            propose_largest_gap_epsilon(&dominated, &gap_candidates(), [1.0, 2.0], [1.0, 1.0]),
            Err(ParetoMetricError::InvalidParetoFront)
        );

        let duplicate_ids = [
            ParetoPoint { trial_id: 1, objectives: [1.0, 9.0] },
            ParetoPoint { trial_id: 1, objectives: [2.0, 7.0] },
        ];
        assert_eq!(
            propose_largest_gap_epsilon(&duplicate_ids, &gap_candidates(), [1.0, 2.0], [1.0, 1.0]),
            Err(ParetoMetricError::DuplicateTrialId)
        );
    }


    fn calibration_samples(scores: &[f64]) -> Vec<PredictionCalibrationSample> {
        scores.iter().map(|score| PredictionCalibrationSample {
            predicted_mean: 0.0,
            predictive_std_dev: 1.0,
            observed_value: *score,
        }).collect()
    }

    #[test]
    fn split_conformal_uses_exact_one_based_finite_sample_rank() {
        let calibration = calibration_samples(&[4.0, 0.5, 2.0, 1.0]);
        let multiplier = fit_split_conformal_multiplier(&calibration, 0.4).unwrap();
        assert_eq!(multiplier.calibration_count, 4);
        assert_eq!(multiplier.order_statistic_rank, 3);
        assert_abs_close(multiplier.multiplier, 2.0, 1e-12);
    }

    #[test]
    fn split_conformal_rejects_too_small_calibration_sets_and_bad_alpha() {
        assert_eq!(
            fit_split_conformal_multiplier(&calibration_samples(&[1.0, 2.0, 3.0]), 0.1),
            Err(ParetoMetricError::InsufficientCalibrationSamples)
        );
        assert_eq!(
            fit_split_conformal_multiplier(&calibration_samples(&[1.0, 2.0]), 0.0),
            Err(ParetoMetricError::InvalidMiscoverage)
        );
        assert_eq!(
            fit_split_conformal_multiplier(&[], 0.1),
            Err(ParetoMetricError::EmptyCalibrationSet)
        );
    }

    #[test]
    fn split_conformal_rejects_invalid_observations_and_uncertainty_scales() {
        assert_eq!(
            fit_split_conformal_multiplier(
                &[PredictionCalibrationSample {
                    predicted_mean: 0.0,
                    predictive_std_dev: 0.0,
                    observed_value: 1.0,
                }],
                0.5
            ),
            Err(ParetoMetricError::InvalidPredictiveScale)
        );
        assert_eq!(
            fit_split_conformal_multiplier(
                &[PredictionCalibrationSample {
                    predicted_mean: f64::NAN,
                    predictive_std_dev: 1.0,
                    observed_value: 1.0,
                }],
                0.5
            ),
            Err(ParetoMetricError::InvalidCalibrationObservation)
        );
        assert_eq!(
            fit_split_conformal_multiplier(
                &[PredictionCalibrationSample {
                    predicted_mean: -f64::MAX,
                    predictive_std_dev: 1.0,
                    observed_value: f64::MAX,
                }],
                0.5
            ),
            Err(ParetoMetricError::NonFiniteCalibrationMetric)
        );
    }

    #[test]
    fn conformal_holdout_reports_coverage_and_gaussian_nll_without_refitting() {
        let calibration = fit_split_conformal_multiplier(&calibration_samples(&[0.5, 1.0, 2.0, 4.0]), 0.4).unwrap();
        let before = calibration;
        let holdout = calibration_samples(&[1.0, 2.0, 3.0]);
        let assessment = assess_conformal_holdout(calibration, &holdout).unwrap();
        assert_eq!(assessment.sample_count, 3);
        assert_eq!(assessment.covered_count, 2);
        assert_abs_close(assessment.empirical_coverage, 2.0 / 3.0, 1e-12);
        assert_abs_close(assessment.mean_absolute_standardized_residual, 2.0, 1e-12);
        let expected_nll = 0.5 * (2.0 * std::f64::consts::PI).ln() + 7.0 / 3.0;
        assert_abs_close(assessment.mean_gaussian_negative_log_likelihood, expected_nll, 1e-12);
        assert_eq!(calibration, before);
    }

    #[test]
    fn conformal_holdout_fails_closed_on_empty_set_or_non_finite_metrics() {
        let calibration = fit_split_conformal_multiplier(&calibration_samples(&[0.5, 1.0, 2.0, 4.0]), 0.4).unwrap();
        assert_eq!(
            assess_conformal_holdout(calibration, &[]),
            Err(ParetoMetricError::EmptyAssessmentSet)
        );
        assert_eq!(
            assess_conformal_holdout(
                calibration,
                &[PredictionCalibrationSample {
                    predicted_mean: 0.0,
                    predictive_std_dev: 1e-300,
                    observed_value: 1.0,
                }]
            ),
            Err(ParetoMetricError::NonFiniteCalibrationMetric)
        );
    }


    fn weighted_calibration_samples(
        scores_and_weights: &[(f64, f64)],
    ) -> Vec<WeightedPredictionCalibrationSample> {
        scores_and_weights.iter().map(|(score, weight)| WeightedPredictionCalibrationSample {
            sample: PredictionCalibrationSample {
                predicted_mean: 0.0,
                predictive_std_dev: 1.0,
                observed_value: *score,
            },
            importance_weight: *weight,
        }).collect()
    }

    #[test]
    fn weighted_conformal_known_answer_includes_test_point_mass() {
        let calibration = weighted_calibration_samples(&[(1.0, 1.0), (2.0, 2.0), (3.0, 1.0)]);
        let result = fit_weighted_split_conformal_multiplier(&calibration, 1.0, 0.25).unwrap();
        assert_abs_close(result.multiplier, 3.0, 1e-12);
        assert_abs_close(result.weighted_effective_sample_size, 8.0 / 3.0, 1e-12);
        assert_abs_close(result.test_point_weight_share, 0.2, 1e-12);
        assert_abs_close(result.finite_quantile_mass_fraction, 0.8, 1e-12);
        assert_eq!(result.calibration_count, 3);
    }

    #[test]
    fn weighted_conformal_is_invariant_to_common_weight_rescaling() {
        let first = fit_weighted_split_conformal_multiplier(
            &weighted_calibration_samples(&[(1.0, 1.0), (2.0, 2.0), (3.0, 1.0)]),
            1.0,
            0.25,
        ).unwrap();
        let scaled = fit_weighted_split_conformal_multiplier(
            &weighted_calibration_samples(&[(1.0, 10.0), (2.0, 20.0), (3.0, 10.0)]),
            10.0,
            0.25,
        ).unwrap();
        assert_abs_close(first.multiplier, scaled.multiplier, 1e-12);
        assert_abs_close(first.weighted_effective_sample_size, scaled.weighted_effective_sample_size, 1e-12);
        assert_abs_close(first.test_point_weight_share, scaled.test_point_weight_share, 1e-12);
        assert_abs_close(first.finite_quantile_mass_fraction, scaled.finite_quantile_mass_fraction, 1e-12);
    }

    #[test]
    fn weighted_conformal_fails_closed_when_test_point_mass_blocks_finite_quantile() {
        let calibration = weighted_calibration_samples(&[(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]);
        assert_eq!(
            fit_weighted_split_conformal_multiplier(&calibration, 4.0, 0.25),
            Err(ParetoMetricError::InsufficientWeightedCalibrationSupport)
        );
        assert_eq!(
            fit_weighted_split_conformal_multiplier(&calibration, 1.0, 0.1),
            Err(ParetoMetricError::InsufficientWeightedCalibrationSupport)
        );
    }

    #[test]
    fn weighted_conformal_rejects_bad_or_zero_total_weights() {
        assert_eq!(
            fit_weighted_split_conformal_multiplier(
                &weighted_calibration_samples(&[(1.0, -1.0), (2.0, 1.0)]),
                1.0,
                0.5
            ),
            Err(ParetoMetricError::InvalidImportanceWeight)
        );
        assert_eq!(
            fit_weighted_split_conformal_multiplier(
                &weighted_calibration_samples(&[(1.0, 0.0), (2.0, 0.0)]),
                1.0,
                0.5
            ),
            Err(ParetoMetricError::ZeroTotalCalibrationWeight)
        );
        assert_eq!(
            fit_weighted_split_conformal_multiplier(
                &weighted_calibration_samples(&[(1.0, 1.0)]),
                f64::NAN,
                0.5
            ),
            Err(ParetoMetricError::InvalidImportanceWeight)
        );
        assert_eq!(
            fit_weighted_split_conformal_multiplier(
                &weighted_calibration_samples(&[(1.0, 1.0)]),
                0.0,
                0.5
            ),
            Err(ParetoMetricError::InvalidImportanceWeight)
        );
    }

    #[test]
    fn weighted_conformal_rejects_underflowing_weight_ratios_and_bad_alpha() {
        assert_eq!(
            fit_weighted_split_conformal_multiplier(
                &weighted_calibration_samples(&[(1.0, f64::MIN_POSITIVE), (2.0, f64::MAX)]),
                f64::MIN_POSITIVE,
                0.5
            ),
            Err(ParetoMetricError::NonFiniteWeightedConformalMetric)
        );
        assert_eq!(
            fit_weighted_split_conformal_multiplier(
                &weighted_calibration_samples(&[(1.0, 1.0)]),
                1.0,
                f64::NAN
            ),
            Err(ParetoMetricError::InvalidMiscoverage)
        );
    }


    #[test]
    fn weighted_conformal_multiplier_can_drive_objective_specific_epsilon_selection() {
        let weighted_q = fit_weighted_split_conformal_multiplier(
            &weighted_calibration_samples(&[(1.0, 1.0), (2.0, 2.0), (3.0, 1.0)]),
            1.0,
            0.25,
        ).unwrap();
        let unweighted_q = fit_split_conformal_multiplier(
            &calibration_samples(&[0.5, 1.0, 2.0, 4.0]),
            0.4,
        ).unwrap();
        assert_abs_close(unweighted_q.multiplier, 2.0, 1e-12);
        assert_abs_close(weighted_q.multiplier, 3.0, 1e-12);

        let candidates = [
            PredictedObjectiveCandidate {
                candidate_id: 10,
                predicted_mean: [2.0, 3.0],
                predictive_std_dev: [0.1, 0.1],
            },
            PredictedObjectiveCandidate {
                candidate_id: 11,
                predicted_mean: [1.8, 3.1],
                predictive_std_dev: [0.5, 0.1],
            },
            PredictedObjectiveCandidate {
                candidate_id: 12,
                predicted_mean: [1.6, 4.0],
                predictive_std_dev: [0.5, 0.5],
            },
        ];
        let proposal = propose_largest_gap_epsilon(
            &gap_front(),
            &candidates,
            [1.0, 2.0],
            [unweighted_q.multiplier, weighted_q.multiplier],
        ).unwrap();
        assert_eq!(proposal.selected_candidate_id, 11);
        assert_abs_close(proposal.objective_0_lower_confidence_bound, 0.8, 1e-12);
        assert_abs_close(proposal.objective_1_upper_confidence_bound, 3.4, 1e-12);
        assert!(proposal.objective_1_upper_confidence_bound <= proposal.epsilon);
    }


    fn post_selection_coverage_samples() -> Vec<PostSelectionCoverageSample> {
        let mut samples = Vec::new();
        // Region 0: selected proposals cover 50%, while unselected observations cover 100%.
        for score in [1.0, 3.0] {
            samples.push(PostSelectionCoverageSample {
                region_id: 0,
                optimizer_selected: true,
                standardized_residual_score: score,
            });
        }
        for _ in 0..4 {
            samples.push(PostSelectionCoverageSample {
                region_id: 0,
                optimizer_selected: false,
                standardized_residual_score: 1.0,
            });
        }
        // Region 1 has the same selection-conditional failure pattern but more easy samples.
        for score in [1.0, 3.0] {
            samples.push(PostSelectionCoverageSample {
                region_id: 1,
                optimizer_selected: true,
                standardized_residual_score: score,
            });
        }
        for _ in 0..6 {
            samples.push(PostSelectionCoverageSample {
                region_id: 1,
                optimizer_selected: false,
                standardized_residual_score: 1.0,
            });
        }
        samples
    }

    #[test]
    fn post_selection_audit_exposes_undercoverage_hidden_by_aggregate_coverage() {
        let audit = audit_post_selection_coverage(&post_selection_coverage_samples(), 2.0, 0.8, 2).unwrap();
        assert_abs_close(audit.overall_coverage, 12.0 / 14.0, 1e-12);
        assert!(audit.overall_coverage > audit.nominal_coverage);
        assert_abs_close(audit.selected_coverage, 0.5, 1e-12);
        assert_abs_close(audit.unselected_coverage, 1.0, 1e-12);
        assert_abs_close(audit.selected_minus_unselected_coverage, -0.5, 1e-12);
        assert_eq!(audit.worst_region_id, 0);
        assert_abs_close(audit.worst_region_coverage, 5.0 / 6.0, 1e-12);
        assert_eq!(audit.regions.iter().map(|r| r.region_id).collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn post_selection_audit_is_invariant_to_input_order() {
        let samples = post_selection_coverage_samples();
        let forward = audit_post_selection_coverage(&samples, 2.0, 0.8, 2).unwrap();
        let reverse_samples: Vec<_> = samples.into_iter().rev().collect();
        let reverse = audit_post_selection_coverage(&reverse_samples, 2.0, 0.8, 2).unwrap();
        assert_eq!(forward, reverse);
    }

    #[test]
    fn post_selection_audit_fails_closed_on_small_regions_and_missing_strata() {
        let small_regions = [
            PostSelectionCoverageSample {
                region_id: 0,
                optimizer_selected: true,
                standardized_residual_score: 1.0,
            },
            PostSelectionCoverageSample {
                region_id: 1,
                optimizer_selected: false,
                standardized_residual_score: 1.0,
            },
        ];
        assert_eq!(
            audit_post_selection_coverage(&small_regions, 2.0, 0.8, 2),
            Err(ParetoMetricError::InsufficientCoverageGroupSample)
        );
        let selected_only = [PostSelectionCoverageSample {
            region_id: 0,
            optimizer_selected: true,
            standardized_residual_score: 1.0,
        }];
        assert_eq!(
            audit_post_selection_coverage(&selected_only, 2.0, 0.8, 1),
            Err(ParetoMetricError::MissingUnselectedCoverage)
        );
        assert_eq!(
            audit_post_selection_coverage(&[], 2.0, 0.8, 1),
            Err(ParetoMetricError::EmptyCoverageAuditSet)
        );
    }

    #[test]
    fn post_selection_audit_rejects_invalid_parameters_and_scores() {
        let samples = post_selection_coverage_samples();
        assert_eq!(
            audit_post_selection_coverage(&samples, f64::NAN, 0.8, 2),
            Err(ParetoMetricError::InvalidCoverageAuditParameter)
        );
        assert_eq!(
            audit_post_selection_coverage(&samples, 2.0, 1.0, 2),
            Err(ParetoMetricError::InvalidCoverageAuditParameter)
        );
        assert_eq!(
            audit_post_selection_coverage(&samples, 2.0, 0.8, 0),
            Err(ParetoMetricError::InvalidCoverageAuditParameter)
        );
        let invalid = [PostSelectionCoverageSample {
            region_id: 0,
            optimizer_selected: true,
            standardized_residual_score: -1.0,
        }, PostSelectionCoverageSample {
            region_id: 0,
            optimizer_selected: false,
            standardized_residual_score: 1.0,
        }];
        assert_eq!(
            audit_post_selection_coverage(&invalid, 2.0, 0.8, 1),
            Err(ParetoMetricError::InvalidResidualScore)
        );
    }

    fn assert_abs_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.15e} differs from expected {expected:.15e} by more than {tolerance:.3e}"
        );
    }
}
