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
            Self::TooFewPointsForSpacing => f.write_str("spacing requires at least two approximation points"),
            Self::NonFiniteMetricResult => f.write_str("front metric overflowed or produced a non-finite result"),
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

#[cfg(test)]
mod tests {
    use super::*;

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
            8.0_f64.sqrt(),
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
            inverted_generational_distance_plus(&reversed_approximation, &reversed_reference).unwrap(),
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

    fn assert_abs_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.15e} differs from expected {expected:.15e} by more than {tolerance:.3e}"
        );
    }
}
