// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bounded recovery metrics for controlled ALife perturbation experiments.
//!
//! These metrics deliberately consume [`crate::ObservatoryReport`] rather than simulator
//! internals. They therefore measure only what the current Genesis event stream establishes:
//! the number of distinct agents observed entering each social tick. They do **not** reinterpret
//! a missing tick as extinction, do not infer exact birth/death timing, and do not claim a causal
//! mechanism for recovery.
//! 
//! A recovery analysis requires contiguous observed-tick coverage from the requested baseline
//! window through the end of the report. This fail-closed rule prevents a truncated event stream
//! from masquerading as a population crash or recovery.

use crate::{ObservatoryReport, ResourcePerturbation};

/// Population-recovery measurements around one controlled perturbation window.
#[derive(Debug, Clone, PartialEq)]
pub struct RecoveryMetrics {
    pub baseline_start_tick: u64,
    pub perturbation_start_tick: u64,
    pub perturbation_end_tick_exclusive: u64,
    pub evaluation_end_tick: u64,
    pub baseline_mean_observed_population: f64,
    pub recovery_fraction: f64,
    pub recovery_target_observed_population: f64,
    pub minimum_fraction_of_baseline_during_perturbation: f64,
    pub minimum_fraction_of_baseline_after_perturbation: f64,
    pub final_fraction_of_baseline: f64,
    /// First post-perturbation tick whose observed population reaches the requested fraction of
    /// the pre-perturbation baseline. `None` means no recovery was observed before the supplied
    /// report ended; it is not an extinction claim.
    pub recovery_tick: Option<u64>,
    pub recovery_latency_ticks: Option<u64>,
    /// Sum over every observed tick from perturbation start through evaluation end of
    /// `max(0, baseline - population) / baseline`. Lower is better. Units are normalized
    /// population-deficit ticks.
    pub normalized_population_deficit_area: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvolvabilityError {
    ZeroBaselineLookback,
    InvalidRecoveryFraction { recovery_fraction: f64 },
    BaselineWindowUnderflow {
        perturbation_start_tick: u64,
        baseline_lookback_ticks: u64,
    },
    EmptyReport,
    EvaluationEndsBeforePerturbation {
        evaluation_end_tick: u64,
        perturbation_end_tick_exclusive: u64,
    },
    MissingObservedTick { tick: u64 },
    ZeroBaselinePopulation,
}

/// Measure observed population recovery around one perturbation.
///
/// `baseline_lookback_ticks` defines the exact half-open pre-perturbation baseline window
/// `[start - lookback, start)`. `recovery_fraction` must be in `(0, 1]`.
pub fn analyze_recovery(
    report: &ObservatoryReport,
    perturbation: ResourcePerturbation,
    baseline_lookback_ticks: u64,
    recovery_fraction: f64,
) -> Result<RecoveryMetrics, EvolvabilityError> {
    if baseline_lookback_ticks == 0 {
        return Err(EvolvabilityError::ZeroBaselineLookback);
    }
    if !recovery_fraction.is_finite() || !(0.0 < recovery_fraction && recovery_fraction <= 1.0) {
        return Err(EvolvabilityError::InvalidRecoveryFraction { recovery_fraction });
    }

    let baseline_start_tick = perturbation
        .start_tick()
        .checked_sub(baseline_lookback_ticks)
        .ok_or(EvolvabilityError::BaselineWindowUnderflow {
            perturbation_start_tick: perturbation.start_tick(),
            baseline_lookback_ticks,
        })?;
    let evaluation_end_tick = report.last_tick.ok_or(EvolvabilityError::EmptyReport)?;
    if evaluation_end_tick < perturbation.end_tick_exclusive() {
        return Err(EvolvabilityError::EvaluationEndsBeforePerturbation {
            evaluation_end_tick,
            perturbation_end_tick_exclusive: perturbation.end_tick_exclusive(),
        });
    }

    // Require a complete baseline and evaluation interval. Absence from the event map is not
    // interpreted as a zero population because the current event stream cannot distinguish an
    // extinct population from a stopped/truncated logger.
    for tick in baseline_start_tick..=evaluation_end_tick {
        require_population(report, tick)?;
    }

    let baseline_total: usize = (baseline_start_tick..perturbation.start_tick())
        .map(|tick| require_population(report, tick))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .sum();
    let baseline_mean = baseline_total as f64 / baseline_lookback_ticks as f64;
    if baseline_mean <= 0.0 {
        return Err(EvolvabilityError::ZeroBaselinePopulation);
    }

    let during_min = (perturbation.start_tick()..perturbation.end_tick_exclusive())
        .map(|tick| require_population(report, tick))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .min()
        .expect("resource perturbations always have non-zero duration");

    let post_start = perturbation.end_tick_exclusive();
    let after_min = (post_start..=evaluation_end_tick)
        .map(|tick| require_population(report, tick))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .min()
        .expect("evaluation end was validated to reach post-perturbation interval");

    let recovery_target = baseline_mean * recovery_fraction;
    let recovery_tick = (post_start..=evaluation_end_tick)
        .find(|&tick| require_population(report, tick).expect("coverage prevalidated") as f64 >= recovery_target);
    let recovery_latency_ticks = recovery_tick.map(|tick| tick - post_start);

    let mut normalized_population_deficit_area = 0.0;
    for tick in perturbation.start_tick()..=evaluation_end_tick {
        let population = require_population(report, tick)? as f64;
        normalized_population_deficit_area += ((baseline_mean - population).max(0.0)) / baseline_mean;
    }

    let final_population = require_population(report, evaluation_end_tick)? as f64;
    Ok(RecoveryMetrics {
        baseline_start_tick,
        perturbation_start_tick: perturbation.start_tick(),
        perturbation_end_tick_exclusive: perturbation.end_tick_exclusive(),
        evaluation_end_tick,
        baseline_mean_observed_population: baseline_mean,
        recovery_fraction,
        recovery_target_observed_population: recovery_target,
        minimum_fraction_of_baseline_during_perturbation: during_min as f64 / baseline_mean,
        minimum_fraction_of_baseline_after_perturbation: after_min as f64 / baseline_mean,
        final_fraction_of_baseline: final_population / baseline_mean,
        recovery_tick,
        recovery_latency_ticks,
        normalized_population_deficit_area,
    })
}

fn require_population(report: &ObservatoryReport, tick: u64) -> Result<usize, EvolvabilityError> {
    report
        .observed_agents_by_tick
        .get(&tick)
        .copied()
        .ok_or(EvolvabilityError::MissingObservedTick { tick })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::PerturbationError;

    fn report(populations: &[(u64, usize)]) -> ObservatoryReport {
        ObservatoryReport {
            event_count: populations.iter().map(|(_, n)| *n).sum(),
            first_tick: populations.first().map(|(tick, _)| *tick),
            last_tick: populations.last().map(|(tick, _)| *tick),
            observed_agents_by_tick: populations.iter().copied().collect::<BTreeMap<_, _>>(),
            agents: BTreeMap::new(),
            lineages: BTreeMap::new(),
            transfers: BTreeMap::new(),
            max_generation: None,
        }
    }

    fn shock() -> Result<ResourcePerturbation, PerturbationError> {
        ResourcePerturbation::new(3, 2, 0.5, 0.0)
    }

    #[test]
    fn measures_recovery_latency_and_deficit_without_inferring_lifecycle_events() {
        let report = report(&[
            (0, 10),
            (1, 10),
            (2, 10),
            (3, 6),
            (4, 4),
            (5, 7),
            (6, 9),
            (7, 10),
        ]);
        let metrics = analyze_recovery(&report, shock().expect("shock"), 3, 0.9).expect("metrics");
        assert_eq!(metrics.baseline_mean_observed_population, 10.0);
        assert_eq!(metrics.minimum_fraction_of_baseline_during_perturbation, 0.4);
        assert_eq!(metrics.minimum_fraction_of_baseline_after_perturbation, 0.7);
        assert_eq!(metrics.recovery_tick, Some(6));
        assert_eq!(metrics.recovery_latency_ticks, Some(1));
        assert_eq!(metrics.final_fraction_of_baseline, 1.0);
        assert!((metrics.normalized_population_deficit_area - 1.4).abs() < 1e-12);
    }

    #[test]
    fn reports_no_observed_recovery_without_calling_absence_extinction() {
        let report = report(&[(0, 10), (1, 10), (2, 10), (3, 8), (4, 7), (5, 7), (6, 8)]);
        let metrics = analyze_recovery(&report, shock().expect("shock"), 3, 0.9).expect("metrics");
        assert_eq!(metrics.recovery_tick, None);
        assert_eq!(metrics.recovery_latency_ticks, None);
        assert_eq!(metrics.final_fraction_of_baseline, 0.8);
    }

    #[test]
    fn missing_tick_fails_closed_instead_of_becoming_zero_population() {
        let report = report(&[(0, 10), (1, 10), (2, 10), (3, 6), (5, 7), (6, 9)]);
        assert_eq!(
            analyze_recovery(&report, shock().expect("shock"), 3, 0.9),
            Err(EvolvabilityError::MissingObservedTick { tick: 4 })
        );
    }

    #[test]
    fn invalid_recovery_contracts_fail_at_the_analysis_boundary() {
        let report = report(&[(0, 10), (1, 10), (2, 10), (3, 6), (4, 4), (5, 8)]);
        assert_eq!(
            analyze_recovery(&report, shock().expect("shock"), 0, 0.9),
            Err(EvolvabilityError::ZeroBaselineLookback)
        );
        assert!(matches!(
            analyze_recovery(&report, shock().expect("shock"), 3, 1.1),
            Err(EvolvabilityError::InvalidRecoveryFraction { .. })
        ));
    }
}
