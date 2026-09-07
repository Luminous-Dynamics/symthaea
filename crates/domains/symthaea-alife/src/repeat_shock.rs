// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Within-condition repeated-shock transfer analysis.
//!
//! This module asks a narrower question than candidate-vs-control recovery comparison:
//!
//! > Under the same condition, did the second matched shock receive a better recovery response
//! > than the first?
//!
//! It requires equal shock shape, equal baseline lookback, equal recovery criterion, equal
//! post-shock evaluation horizon, and non-overlapping evaluation windows. The result is directional
//! evidence only; it does not prove learning, evolution, or a specific mechanism caused any change.

use crate::{RecoveryMetrics, ResourcePerturbation};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatShockLatencyV1 {
    BothRecovered {
        first_latency_ticks: u64,
        second_latency_ticks: u64,
        /// Positive means the second shock recovered sooner.
        latency_advantage_ticks: i128,
    },
    SecondOnlyRecovered,
    FirstOnlyRecovered,
    NeitherRecovered,
}

/// Directional Pareto classification across the predeclared recovery dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatShockTransferVerdictV1 {
    /// At least one dimension improved and no dimension worsened.
    ParetoImproved,
    /// Some dimensions improved while others worsened.
    Mixed,
    /// No directional difference in the selected dimensions.
    NoDirectionalChange,
    /// At least one dimension worsened and no dimension improved.
    ParetoWorse,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RepeatShockTransferV1 {
    pub first_baseline_mean_observed_population: f64,
    pub second_baseline_mean_observed_population: f64,
    /// Positive means the second shock retained more of its own baseline during the perturbation.
    pub minimum_during_fraction_delta: f64,
    /// Positive means the second shock retained more of its own baseline after the perturbation.
    pub minimum_after_fraction_delta: f64,
    /// Positive means the second shock ended the matched horizon at a larger fraction of baseline.
    pub final_fraction_delta: f64,
    /// Positive means the second shock accumulated less normalized population deficit.
    pub deficit_area_advantage: f64,
    pub latency: RepeatShockLatencyV1,
    pub verdict: RepeatShockTransferVerdictV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatShockErrorV1 {
    ShockDurationMismatch {
        first_duration: u64,
        second_duration: u64,
    },
    ShockMultiplierMismatch,
    ShockDeltaMismatch,
    MetricShockBoundaryMismatch,
    InvalidMetricBaselineBoundary,
    BaselineLookbackMismatch {
        first_lookback: u64,
        second_lookback: u64,
    },
    RecoveryFractionMismatch,
    InvalidMetricEvaluationBoundary,
    EvaluationHorizonMismatch {
        first_post_shock_ticks: u64,
        second_post_shock_ticks: u64,
    },
    SecondShockOverlapsFirstEvaluation {
        first_evaluation_end_tick: u64,
        second_shock_start_tick: u64,
    },
}

/// Compare two matched shocks within one condition.
pub fn compare_repeated_shocks(
    first_shock: ResourcePerturbation,
    first: &RecoveryMetrics,
    second_shock: ResourcePerturbation,
    second: &RecoveryMetrics,
) -> Result<RepeatShockTransferV1, RepeatShockErrorV1> {
    if first_shock.duration_ticks() != second_shock.duration_ticks() {
        return Err(RepeatShockErrorV1::ShockDurationMismatch {
            first_duration: first_shock.duration_ticks(),
            second_duration: second_shock.duration_ticks(),
        });
    }
    if first_shock.multiplier().to_bits() != second_shock.multiplier().to_bits() {
        return Err(RepeatShockErrorV1::ShockMultiplierMismatch);
    }
    if first_shock.delta().to_bits() != second_shock.delta().to_bits() {
        return Err(RepeatShockErrorV1::ShockDeltaMismatch);
    }

    if first.perturbation_start_tick != first_shock.start_tick()
        || first.perturbation_end_tick_exclusive != first_shock.end_tick_exclusive()
        || second.perturbation_start_tick != second_shock.start_tick()
        || second.perturbation_end_tick_exclusive != second_shock.end_tick_exclusive()
    {
        return Err(RepeatShockErrorV1::MetricShockBoundaryMismatch);
    }

    let first_lookback = first
        .perturbation_start_tick
        .checked_sub(first.baseline_start_tick)
        .ok_or(RepeatShockErrorV1::InvalidMetricBaselineBoundary)?;
    let second_lookback = second
        .perturbation_start_tick
        .checked_sub(second.baseline_start_tick)
        .ok_or(RepeatShockErrorV1::InvalidMetricBaselineBoundary)?;
    if first_lookback != second_lookback {
        return Err(RepeatShockErrorV1::BaselineLookbackMismatch {
            first_lookback,
            second_lookback,
        });
    }
    if first.recovery_fraction.to_bits() != second.recovery_fraction.to_bits() {
        return Err(RepeatShockErrorV1::RecoveryFractionMismatch);
    }

    let first_post_shock_ticks = first
        .evaluation_end_tick
        .checked_sub(first.perturbation_end_tick_exclusive)
        .and_then(|delta| delta.checked_add(1))
        .ok_or(RepeatShockErrorV1::InvalidMetricEvaluationBoundary)?;
    let second_post_shock_ticks = second
        .evaluation_end_tick
        .checked_sub(second.perturbation_end_tick_exclusive)
        .and_then(|delta| delta.checked_add(1))
        .ok_or(RepeatShockErrorV1::InvalidMetricEvaluationBoundary)?;
    if first_post_shock_ticks != second_post_shock_ticks {
        return Err(RepeatShockErrorV1::EvaluationHorizonMismatch {
            first_post_shock_ticks,
            second_post_shock_ticks,
        });
    }
    if second_shock.start_tick() <= first.evaluation_end_tick {
        return Err(RepeatShockErrorV1::SecondShockOverlapsFirstEvaluation {
            first_evaluation_end_tick: first.evaluation_end_tick,
            second_shock_start_tick: second_shock.start_tick(),
        });
    }

    let minimum_during_fraction_delta = second.minimum_fraction_of_baseline_during_perturbation
        - first.minimum_fraction_of_baseline_during_perturbation;
    let minimum_after_fraction_delta = second.minimum_fraction_of_baseline_after_perturbation
        - first.minimum_fraction_of_baseline_after_perturbation;
    let final_fraction_delta = second.final_fraction_of_baseline - first.final_fraction_of_baseline;
    let deficit_area_advantage =
        first.normalized_population_deficit_area - second.normalized_population_deficit_area;

    let latency = match (first.recovery_latency_ticks, second.recovery_latency_ticks) {
        (Some(first_latency_ticks), Some(second_latency_ticks)) => {
            RepeatShockLatencyV1::BothRecovered {
                first_latency_ticks,
                second_latency_ticks,
                latency_advantage_ticks: i128::from(first_latency_ticks)
                    - i128::from(second_latency_ticks),
            }
        }
        (None, Some(_)) => RepeatShockLatencyV1::SecondOnlyRecovered,
        (Some(_), None) => RepeatShockLatencyV1::FirstOnlyRecovered,
        (None, None) => RepeatShockLatencyV1::NeitherRecovered,
    };

    let mut any_better = false;
    let mut any_worse = false;
    record_direction(
        minimum_during_fraction_delta,
        &mut any_better,
        &mut any_worse,
    );
    record_direction(
        minimum_after_fraction_delta,
        &mut any_better,
        &mut any_worse,
    );
    record_direction(final_fraction_delta, &mut any_better, &mut any_worse);
    record_direction(deficit_area_advantage, &mut any_better, &mut any_worse);
    match latency {
        RepeatShockLatencyV1::BothRecovered {
            latency_advantage_ticks,
            ..
        } => {
            if latency_advantage_ticks > 0 {
                any_better = true;
            } else if latency_advantage_ticks < 0 {
                any_worse = true;
            }
        }
        RepeatShockLatencyV1::SecondOnlyRecovered => any_better = true,
        RepeatShockLatencyV1::FirstOnlyRecovered => any_worse = true,
        RepeatShockLatencyV1::NeitherRecovered => {}
    }

    let verdict = match (any_better, any_worse) {
        (true, false) => RepeatShockTransferVerdictV1::ParetoImproved,
        (true, true) => RepeatShockTransferVerdictV1::Mixed,
        (false, false) => RepeatShockTransferVerdictV1::NoDirectionalChange,
        (false, true) => RepeatShockTransferVerdictV1::ParetoWorse,
    };

    Ok(RepeatShockTransferV1 {
        first_baseline_mean_observed_population: first.baseline_mean_observed_population,
        second_baseline_mean_observed_population: second.baseline_mean_observed_population,
        minimum_during_fraction_delta,
        minimum_after_fraction_delta,
        final_fraction_delta,
        deficit_area_advantage,
        latency,
        verdict,
    })
}

fn record_direction(value: f64, any_better: &mut bool, any_worse: &mut bool) {
    if value > 0.0 {
        *any_better = true;
    } else if value < 0.0 {
        *any_worse = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shock(start: u64) -> ResourcePerturbation {
        ResourcePerturbation::new(start, 10, 0.7, -0.05).expect("valid shock")
    }

    fn metrics(start: u64, recovery_latency_ticks: Option<u64>) -> RecoveryMetrics {
        let end = start + 10;
        RecoveryMetrics {
            baseline_start_tick: start - 20,
            perturbation_start_tick: start,
            perturbation_end_tick_exclusive: end,
            evaluation_end_tick: end + 49,
            baseline_mean_observed_population: 20.0,
            recovery_fraction: 0.9,
            recovery_target_observed_population: 18.0,
            minimum_fraction_of_baseline_during_perturbation: 0.6,
            minimum_fraction_of_baseline_after_perturbation: 0.7,
            final_fraction_of_baseline: 0.9,
            recovery_tick: recovery_latency_ticks.map(|latency| end + latency),
            recovery_latency_ticks,
            normalized_population_deficit_area: 8.0,
        }
    }

    #[test]
    fn pareto_improvement_requires_no_selected_dimension_to_worsen() {
        let first_shock = shock(100);
        let second_shock = shock(300);
        let first = metrics(100, Some(20));
        let mut second = metrics(300, Some(10));
        second.minimum_fraction_of_baseline_during_perturbation = 0.7;
        second.minimum_fraction_of_baseline_after_perturbation = 0.8;
        second.final_fraction_of_baseline = 1.0;
        second.normalized_population_deficit_area = 5.0;

        let transfer = compare_repeated_shocks(first_shock, &first, second_shock, &second)
            .expect("matched repeated shocks");
        assert_eq!(
            transfer.verdict,
            RepeatShockTransferVerdictV1::ParetoImproved
        );
        assert_eq!(
            transfer.latency,
            RepeatShockLatencyV1::BothRecovered {
                first_latency_ticks: 20,
                second_latency_ticks: 10,
                latency_advantage_ticks: 10,
            }
        );
        assert!(transfer.deficit_area_advantage > 0.0);
    }

    #[test]
    fn mixed_outcome_is_not_collapsed_into_improvement() {
        let first_shock = shock(100);
        let second_shock = shock(300);
        let first = metrics(100, Some(20));
        let mut second = metrics(300, Some(10));
        second.minimum_fraction_of_baseline_during_perturbation = 0.5;
        second.normalized_population_deficit_area = 5.0;

        let transfer = compare_repeated_shocks(first_shock, &first, second_shock, &second)
            .expect("matched repeated shocks");
        assert_eq!(transfer.verdict, RepeatShockTransferVerdictV1::Mixed);
    }

    #[test]
    fn unequal_post_shock_horizons_fail_closed() {
        let first_shock = shock(100);
        let second_shock = shock(300);
        let first = metrics(100, Some(20));
        let mut second = metrics(300, Some(20));
        second.evaluation_end_tick += 1;

        assert!(matches!(
            compare_repeated_shocks(first_shock, &first, second_shock, &second),
            Err(RepeatShockErrorV1::EvaluationHorizonMismatch { .. })
        ));
    }

    #[test]
    fn second_shock_cannot_contaminate_first_recovery_window() {
        let first_shock = shock(100);
        let second_shock = shock(150);
        let first = metrics(100, Some(20));
        let second = metrics(150, Some(20));

        assert!(matches!(
            compare_repeated_shocks(first_shock, &first, second_shock, &second),
            Err(RepeatShockErrorV1::SecondShockOverlapsFirstEvaluation { .. })
        ));
    }
}
