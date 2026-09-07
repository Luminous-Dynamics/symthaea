// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Matched comparison of ALife recovery experiments.
//!
//! A single recovery trajectory is not enough to establish that evolution, learning, or another
//! mechanism improved resilience. This module compares two already-qualified
//! [`crate::RecoveryMetrics`] reports only when their experimental boundaries match exactly.
//! It intentionally does not decide *why* one condition recovered better.

use crate::RecoveryMetrics;

/// Recovery-latency comparison without treating a missing recovery as an infinite number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryLatencyComparison {
    BothRecovered {
        candidate_ticks: u64,
        control_ticks: u64,
        /// `control - candidate`; positive means the candidate recovered sooner.
        latency_advantage_ticks: i128,
    },
    CandidateOnlyRecovered,
    ControlOnlyRecovered,
    NeitherRecovered,
}

/// Normalized candidate-minus-control comparison. Positive advantage values mean the candidate
/// did better on that dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct RecoveryComparison {
    pub minimum_during_advantage: f64,
    pub minimum_after_advantage: f64,
    pub final_population_advantage: f64,
    /// `control deficit area - candidate deficit area`; positive means less accumulated deficit
    /// in the candidate condition.
    pub deficit_area_advantage: f64,
    pub latency: RecoveryLatencyComparison,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryComparisonError {
    IncomparableExperiment { field: &'static str },
}

/// Compare a candidate recovery result with its matched control.
///
/// Both metrics must share the exact baseline window, perturbation window, evaluation end, and
/// recovery threshold. Baseline population sizes may differ: the compared population quantities
/// are already normalized to each condition's own pre-perturbation baseline.
pub fn compare_recovery(
    candidate: &RecoveryMetrics,
    control: &RecoveryMetrics,
) -> Result<RecoveryComparison, RecoveryComparisonError> {
    require_equal(
        candidate.baseline_start_tick == control.baseline_start_tick,
        "baseline_start_tick",
    )?;
    require_equal(
        candidate.perturbation_start_tick == control.perturbation_start_tick,
        "perturbation_start_tick",
    )?;
    require_equal(
        candidate.perturbation_end_tick_exclusive == control.perturbation_end_tick_exclusive,
        "perturbation_end_tick_exclusive",
    )?;
    require_equal(
        candidate.evaluation_end_tick == control.evaluation_end_tick,
        "evaluation_end_tick",
    )?;
    require_equal(
        candidate.recovery_fraction == control.recovery_fraction,
        "recovery_fraction",
    )?;

    let latency = match (
        candidate.recovery_latency_ticks,
        control.recovery_latency_ticks,
    ) {
        (Some(candidate_ticks), Some(control_ticks)) => RecoveryLatencyComparison::BothRecovered {
            candidate_ticks,
            control_ticks,
            latency_advantage_ticks: i128::from(control_ticks) - i128::from(candidate_ticks),
        },
        (Some(_), None) => RecoveryLatencyComparison::CandidateOnlyRecovered,
        (None, Some(_)) => RecoveryLatencyComparison::ControlOnlyRecovered,
        (None, None) => RecoveryLatencyComparison::NeitherRecovered,
    };

    Ok(RecoveryComparison {
        minimum_during_advantage: candidate.minimum_fraction_of_baseline_during_perturbation
            - control.minimum_fraction_of_baseline_during_perturbation,
        minimum_after_advantage: candidate.minimum_fraction_of_baseline_after_perturbation
            - control.minimum_fraction_of_baseline_after_perturbation,
        final_population_advantage: candidate.final_fraction_of_baseline
            - control.final_fraction_of_baseline,
        deficit_area_advantage: control.normalized_population_deficit_area
            - candidate.normalized_population_deficit_area,
        latency,
    })
}

fn require_equal(equal: bool, field: &'static str) -> Result<(), RecoveryComparisonError> {
    if equal {
        Ok(())
    } else {
        Err(RecoveryComparisonError::IncomparableExperiment { field })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(
        minimum_during: f64,
        minimum_after: f64,
        final_fraction: f64,
        deficit_area: f64,
        recovery_latency_ticks: Option<u64>,
    ) -> RecoveryMetrics {
        RecoveryMetrics {
            baseline_start_tick: 0,
            perturbation_start_tick: 10,
            perturbation_end_tick_exclusive: 15,
            evaluation_end_tick: 30,
            baseline_mean_observed_population: 100.0,
            recovery_fraction: 0.9,
            recovery_target_observed_population: 90.0,
            minimum_fraction_of_baseline_during_perturbation: minimum_during,
            minimum_fraction_of_baseline_after_perturbation: minimum_after,
            final_fraction_of_baseline: final_fraction,
            recovery_tick: recovery_latency_ticks.map(|latency| 15 + latency),
            recovery_latency_ticks,
            normalized_population_deficit_area: deficit_area,
        }
    }

    #[test]
    fn positive_advantages_consistently_mean_candidate_is_better() {
        let candidate = metrics(0.6, 0.7, 1.0, 2.0, Some(4));
        let control = metrics(0.4, 0.5, 0.8, 4.5, Some(9));
        let comparison = compare_recovery(&candidate, &control).expect("matched comparison");
        assert!((comparison.minimum_during_advantage - 0.2).abs() < 1e-12);
        assert!((comparison.minimum_after_advantage - 0.2).abs() < 1e-12);
        assert!((comparison.final_population_advantage - 0.2).abs() < 1e-12);
        assert!((comparison.deficit_area_advantage - 2.5).abs() < 1e-12);
        assert_eq!(
            comparison.latency,
            RecoveryLatencyComparison::BothRecovered {
                candidate_ticks: 4,
                control_ticks: 9,
                latency_advantage_ticks: 5,
            }
        );
    }

    #[test]
    fn candidate_only_recovery_is_not_encoded_as_infinite_control_latency() {
        let candidate = metrics(0.6, 0.7, 0.95, 2.0, Some(4));
        let control = metrics(0.4, 0.5, 0.7, 5.0, None);
        let comparison = compare_recovery(&candidate, &control).expect("matched comparison");
        assert_eq!(
            comparison.latency,
            RecoveryLatencyComparison::CandidateOnlyRecovered
        );
    }

    #[test]
    fn mismatched_experiment_boundaries_fail_closed() {
        let candidate = metrics(0.6, 0.7, 1.0, 2.0, Some(4));
        let mut control = metrics(0.4, 0.5, 0.8, 4.5, Some(9));
        control.evaluation_end_tick = 31;
        assert_eq!(
            compare_recovery(&candidate, &control),
            Err(RecoveryComparisonError::IncomparableExperiment {
                field: "evaluation_end_tick"
            })
        );
    }
}
