// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Delayed-label outcome validation for feature-shift alarms.
//!
//! A feature-shift alarm is not, by itself, evidence of consequential model or
//! system degradation. This module compares a predeclared outcome window before
//! an alarm with a post-alarm window while recording when each outcome became
//! available. It intentionally reports temporal outcome association, not
//! causation. Missing labels, incomplete windows, insufficient history, and
//! overlapping alarm windows remain explicit unresolved states.

use std::collections::{BTreeMap, BTreeSet};

/// Defines which direction of a measured outcome is operationally worse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutcomeDirection {
    /// Larger observed values represent worse outcomes, such as error or cost.
    HigherIsWorse,
    /// Smaller observed values represent worse outcomes, such as accuracy or uptime.
    LowerIsWorse,
}

/// Frozen evaluation settings. Configure these before examining the evaluation stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DelayedOutcomeConfig {
    /// Number of consecutive stream observations in each outcome window.
    pub window_size: u64,
    /// Minimum labels required in each window before comparing means.
    pub min_labeled_per_window: usize,
    /// Minimum fraction of labels required in each window, in (0, 1].
    pub min_labeled_fraction: f64,
    /// Minimum worse-oriented mean change that counts as material.
    pub minimum_effect: f64,
    /// Direction in which observed outcomes are considered worse.
    pub direction: OutcomeDirection,
}

impl DelayedOutcomeConfig {
    fn validate(self) -> Result<(), DelayedOutcomeError> {
        if self.window_size == 0 {
            return Err(DelayedOutcomeError::InvalidWindowSize);
        }
        let Some(minimum) = u64::try_from(self.min_labeled_per_window).ok() else {
            return Err(DelayedOutcomeError::InvalidMinimumLabelCount);
        };
        if minimum == 0 || minimum > self.window_size {
            return Err(DelayedOutcomeError::InvalidMinimumLabelCount);
        }
        if !self.min_labeled_fraction.is_finite()
            || self.min_labeled_fraction <= 0.0
            || self.min_labeled_fraction > 1.0
        {
            return Err(DelayedOutcomeError::InvalidMinimumLabelFraction);
        }
        if !self.minimum_effect.is_finite() || self.minimum_effect < 0.0 {
            return Err(DelayedOutcomeError::InvalidMinimumEffect);
        }
        Ok(())
    }
}

/// Input-contract failures. No failure is silently converted to a no-drift result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelayedOutcomeError {
    InvalidWindowSize,
    InvalidMinimumLabelCount,
    InvalidMinimumLabelFraction,
    InvalidMinimumEffect,
    NonSequentialObservation,
    StreamIndexOverflow,
    AlarmForUnknownObservation,
    AlarmNotAtCurrentStreamPosition,
    AlarmAfterOutcome,
    DuplicateAlarm,
    UnknownAlarm,
    OutcomeForUnknownObservation,
    DuplicateOutcome,
    OutcomeArrivedBeforeObservation,
    FutureLabelArrival,
    HistoricalLabelArrival,
    NonFiniteOutcome,
}

/// A single outcome window. The denominator is the configured window length,
/// not the number of labels that happened to arrive.
#[derive(Clone, Debug, PartialEq)]
pub struct OutcomeWindowSummary {
    pub start_inclusive: u64,
    pub end_exclusive: u64,
    pub expected_observations: usize,
    pub labeled_observations: usize,
    pub label_coverage_fraction: f64,
    /// Mean of the outcomes available by the monitor's current stream position.
    pub mean_outcome: Option<f64>,
}

/// Status of the observed change following an alarm.
///
/// These labels describe association in two windows. They do not establish that
/// the alarm caused the change or that the change has the same cause as the shift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutcomeAssociation {
    WindowNotComplete,
    MissingBaselineHistory,
    ContaminatedByAnotherAlarm,
    InsufficientLabelCoverage,
    NumericalOverflow,
    Deteriorated,
    NoMaterialChange,
    Improved,
}

/// Auditable result for one alarm. The worse-oriented change is positive when
/// the post-window mean moved in the configured worse direction.
#[derive(Clone, Debug, PartialEq)]
pub struct DelayedOutcomeAssessment {
    pub alarm_index: u64,
    pub baseline: OutcomeWindowSummary,
    pub post_alarm: OutcomeWindowSummary,
    pub worse_oriented_change: Option<f64>,
    pub association: OutcomeAssociation,
}

#[derive(Clone, Copy, Debug)]
struct OutcomeReceipt {
    value: f64,
    arrival_index: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Observation {
    outcome: Option<OutcomeReceipt>,
}

/// Tracks stream positions, shift alarms, and outcome-label arrival times.
///
/// Call observe once per input item in stream order. Call raise_alarm at the
/// index where the feature-shift alarm occurred. When an outcome label becomes
/// available, call record_outcome with both its original observation index and
/// the current stream index at which the label arrived. A later assess call
/// may change from unresolved to resolved as labels arrive.
#[derive(Clone, Debug)]
pub struct DelayedOutcomeMonitor {
    config: DelayedOutcomeConfig,
    observations: BTreeMap<u64, Observation>,
    alarms: BTreeSet<u64>,
    first_index: Option<u64>,
    last_index: Option<u64>,
}

impl DelayedOutcomeMonitor {
    pub fn new(config: DelayedOutcomeConfig) -> Result<Self, DelayedOutcomeError> {
        config.validate()?;
        Ok(Self {
            config,
            observations: BTreeMap::new(),
            alarms: BTreeSet::new(),
            first_index: None,
            last_index: None,
        })
    }

    /// Register one observed stream item. Indices must be consecutive and increasing.
    pub fn observe(&mut self, index: u64) -> Result<(), DelayedOutcomeError> {
        if let Some(last) = self.last_index {
            let expected = last
                .checked_add(1)
                .ok_or(DelayedOutcomeError::StreamIndexOverflow)?;
            if index != expected {
                return Err(DelayedOutcomeError::NonSequentialObservation);
            }
        } else {
            self.first_index = Some(index);
        }
        self.observations.insert(index, Observation::default());
        self.last_index = Some(index);
        Ok(())
    }

    /// Record an alarm at the current stream position, before consuming that
    /// position's outcome label.
    pub fn raise_alarm(&mut self, alarm_index: u64) -> Result<(), DelayedOutcomeError> {
        if !self.observations.contains_key(&alarm_index) {
            return Err(DelayedOutcomeError::AlarmForUnknownObservation);
        }
        if self.last_index != Some(alarm_index) {
            return Err(DelayedOutcomeError::AlarmNotAtCurrentStreamPosition);
        }
        if self
            .observations
            .get(&alarm_index)
            .and_then(|observation| observation.outcome)
            .is_some()
        {
            return Err(DelayedOutcomeError::AlarmAfterOutcome);
        }
        if !self.alarms.insert(alarm_index) {
            return Err(DelayedOutcomeError::DuplicateAlarm);
        }
        Ok(())
    }

    /// Attach an outcome to the original observation when its label arrives.
    ///
    /// arrival_index must equal the current observed stream position. This
    /// prevents future or retrospectively backdated labels in online replay.
    pub fn record_outcome(
        &mut self,
        observation_index: u64,
        arrival_index: u64,
        value: f64,
    ) -> Result<(), DelayedOutcomeError> {
        if !value.is_finite() {
            return Err(DelayedOutcomeError::NonFiniteOutcome);
        }
        if arrival_index < observation_index {
            return Err(DelayedOutcomeError::OutcomeArrivedBeforeObservation);
        }
        let Some(latest_index) = self.last_index else {
            return Err(DelayedOutcomeError::OutcomeForUnknownObservation);
        };
        if arrival_index > latest_index {
            return Err(DelayedOutcomeError::FutureLabelArrival);
        }
        if arrival_index < latest_index {
            return Err(DelayedOutcomeError::HistoricalLabelArrival);
        }
        let observation = self
            .observations
            .get_mut(&observation_index)
            .ok_or(DelayedOutcomeError::OutcomeForUnknownObservation)?;
        if observation.outcome.is_some() {
            return Err(DelayedOutcomeError::DuplicateOutcome);
        }
        observation.outcome = Some(OutcomeReceipt {
            value,
            arrival_index,
        });
        Ok(())
    }

    /// Assess one alarm using fixed pre/post windows and the labels available now.
    ///
    /// The baseline window is [alarm_index - window_size, alarm_index) and the
    /// post-alarm window is [alarm_index, alarm_index + window_size). Another
    /// alarm inside either window makes the comparison contaminated.
    pub fn assess(
        &self,
        alarm_index: u64,
    ) -> Result<DelayedOutcomeAssessment, DelayedOutcomeError> {
        if !self.alarms.contains(&alarm_index) {
            return Err(DelayedOutcomeError::UnknownAlarm);
        }

        let window_size = self.config.window_size;
        let baseline_start = alarm_index.checked_sub(window_size);
        let baseline_start_for_summary = baseline_start.unwrap_or(0);
        let post_end = alarm_index
            .checked_add(window_size)
            .ok_or(DelayedOutcomeError::StreamIndexOverflow)?;

        let expected =
            usize::try_from(window_size).map_err(|_| DelayedOutcomeError::InvalidWindowSize)?;
        let baseline = self.summarize_window(baseline_start_for_summary, alarm_index, expected);
        let post_alarm = self.summarize_window(alarm_index, post_end, expected);
        let latest_index = self.last_index.unwrap_or(alarm_index);

        let post_end_index = post_end - 1;
        let window_complete = latest_index >= post_end_index;
        let baseline_history_available = baseline_start.is_some()
            && self
                .first_index
                .map(|first| first <= baseline_start_for_summary)
                .unwrap_or(false);

        let another_alarm_in_windows = self
            .alarms
            .range(baseline_start_for_summary..post_end)
            .any(|index| *index != alarm_index);

        let mut assessment = DelayedOutcomeAssessment {
            alarm_index,
            baseline,
            post_alarm,
            worse_oriented_change: None,
            association: OutcomeAssociation::WindowNotComplete,
        };

        if !window_complete {
            return Ok(assessment);
        }
        if !baseline_history_available {
            assessment.association = OutcomeAssociation::MissingBaselineHistory;
            return Ok(assessment);
        }
        if another_alarm_in_windows {
            assessment.association = OutcomeAssociation::ContaminatedByAnotherAlarm;
            return Ok(assessment);
        }
        if !self.has_sufficient_labels(&assessment.baseline)
            || !self.has_sufficient_labels(&assessment.post_alarm)
        {
            assessment.association = OutcomeAssociation::InsufficientLabelCoverage;
            return Ok(assessment);
        }

        let (Some(baseline_mean), Some(post_mean)) = (
            assessment.baseline.mean_outcome,
            assessment.post_alarm.mean_outcome,
        ) else {
            assessment.association = OutcomeAssociation::NumericalOverflow;
            return Ok(assessment);
        };

        let worse_oriented_change = match self.config.direction {
            OutcomeDirection::HigherIsWorse => post_mean - baseline_mean,
            OutcomeDirection::LowerIsWorse => baseline_mean - post_mean,
        };
        if !worse_oriented_change.is_finite() {
            assessment.association = OutcomeAssociation::NumericalOverflow;
            return Ok(assessment);
        }
        assessment.worse_oriented_change = Some(worse_oriented_change);
        assessment.association = if worse_oriented_change > self.config.minimum_effect {
            OutcomeAssociation::Deteriorated
        } else if worse_oriented_change < -self.config.minimum_effect {
            OutcomeAssociation::Improved
        } else {
            OutcomeAssociation::NoMaterialChange
        };
        Ok(assessment)
    }

    fn has_sufficient_labels(&self, summary: &OutcomeWindowSummary) -> bool {
        summary.labeled_observations >= self.config.min_labeled_per_window
            && summary.label_coverage_fraction >= self.config.min_labeled_fraction
    }

    fn summarize_window(
        &self,
        start_inclusive: u64,
        end_exclusive: u64,
        expected_observations: usize,
    ) -> OutcomeWindowSummary {
        let mut labeled_observations = 0_usize;
        let mut sum = 0.0_f64;
        let mut numerical_overflow = false;

        for observation in self
            .observations
            .range(start_inclusive..end_exclusive)
            .map(|(_, row)| row)
        {
            if let Some(receipt) = observation.outcome {
                if self
                    .last_index
                    .is_some_and(|last| receipt.arrival_index <= last)
                {
                    labeled_observations += 1;
                    if !numerical_overflow {
                        let candidate = sum + receipt.value;
                        if candidate.is_finite() {
                            sum = candidate;
                        } else {
                            numerical_overflow = true;
                        }
                    }
                }
            }
        }

        let mean_outcome = if labeled_observations == 0 || numerical_overflow {
            None
        } else {
            let mean = sum / labeled_observations as f64;
            mean.is_finite().then_some(mean)
        };
        let label_coverage_fraction = labeled_observations as f64 / expected_observations as f64;

        OutcomeWindowSummary {
            start_inclusive,
            end_exclusive,
            expected_observations,
            labeled_observations,
            label_coverage_fraction,
            mean_outcome,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn monitor(direction: OutcomeDirection) -> DelayedOutcomeMonitor {
        DelayedOutcomeMonitor::new(DelayedOutcomeConfig {
            window_size: 3,
            min_labeled_per_window: 2,
            min_labeled_fraction: 0.66,
            minimum_effect: 0.5,
            direction,
        })
        .unwrap()
    }

    fn observe_through(monitor: &mut DelayedOutcomeMonitor, end_exclusive: u64) {
        let start = monitor.last_index.map(|index| index + 1).unwrap_or(0);
        for index in start..end_exclusive {
            monitor.observe(index).unwrap();
        }
    }

    fn alarm_at_five(monitor: &mut DelayedOutcomeMonitor) {
        observe_through(monitor, 6);
        monitor.raise_alarm(5).unwrap();
        observe_through(monitor, 10);
    }

    #[test]
    fn delayed_labels_move_assessment_from_unresolved_to_deteriorated() {
        let mut monitor = monitor(OutcomeDirection::HigherIsWorse);
        alarm_at_five(&mut monitor);
        let before_labels = monitor.assess(5).unwrap();
        assert_eq!(
            before_labels.association,
            OutcomeAssociation::InsufficientLabelCoverage
        );

        for index in 2..5 {
            monitor.record_outcome(index, 9, 1.0).unwrap();
        }
        for index in 5..8 {
            monitor.record_outcome(index, 9, 3.0).unwrap();
        }

        let after_labels = monitor.assess(5).unwrap();
        assert_eq!(after_labels.association, OutcomeAssociation::Deteriorated);
        assert_eq!(after_labels.worse_oriented_change, Some(2.0));
        assert_eq!(after_labels.baseline.label_coverage_fraction, 1.0);
        assert_eq!(after_labels.post_alarm.label_coverage_fraction, 1.0);
    }

    #[test]
    fn direction_is_explicit_for_metrics_where_lower_is_worse() {
        let mut monitor = monitor(OutcomeDirection::LowerIsWorse);
        alarm_at_five(&mut monitor);
        for index in 2..5 {
            monitor.record_outcome(index, 9, 5.0).unwrap();
        }
        for index in 5..8 {
            monitor.record_outcome(index, 9, 2.0).unwrap();
        }
        let assessment = monitor.assess(5).unwrap();
        assert_eq!(assessment.association, OutcomeAssociation::Deteriorated);
        assert_eq!(assessment.worse_oriented_change, Some(3.0));
    }

    #[test]
    fn incomplete_post_window_is_not_reported_as_no_change() {
        let mut monitor = monitor(OutcomeDirection::HigherIsWorse);
        observe_through(&mut monitor, 6);
        monitor.raise_alarm(5).unwrap();
        monitor.observe(6).unwrap();
        assert_eq!(
            monitor.assess(5).unwrap().association,
            OutcomeAssociation::WindowNotComplete
        );
    }

    #[test]
    fn overlapping_alarm_windows_remain_unresolved() {
        let mut monitor = monitor(OutcomeDirection::HigherIsWorse);
        observe_through(&mut monitor, 5);
        monitor.raise_alarm(4).unwrap();
        monitor.observe(5).unwrap();
        monitor.raise_alarm(5).unwrap();
        observe_through(&mut monitor, 10);
        assert_eq!(
            monitor.assess(4).unwrap().association,
            OutcomeAssociation::ContaminatedByAnotherAlarm
        );
    }

    #[test]
    fn insufficient_history_is_not_silently_treated_as_a_baseline() {
        let mut monitor = monitor(OutcomeDirection::HigherIsWorse);
        observe_through(&mut monitor, 2);
        monitor.raise_alarm(1).unwrap();
        observe_through(&mut monitor, 8);
        assert_eq!(
            monitor.assess(1).unwrap().association,
            OutcomeAssociation::MissingBaselineHistory
        );
    }

    #[test]
    fn rejects_future_duplicate_nonfinite_and_out_of_order_receipts() {
        let mut monitor = monitor(OutcomeDirection::HigherIsWorse);
        alarm_at_five(&mut monitor);

        assert_eq!(
            monitor.record_outcome(5, 10, 1.0),
            Err(DelayedOutcomeError::FutureLabelArrival)
        );
        assert_eq!(
            monitor.record_outcome(5, 4, 1.0),
            Err(DelayedOutcomeError::OutcomeArrivedBeforeObservation)
        );
        assert_eq!(
            monitor.record_outcome(5, 8, 1.0),
            Err(DelayedOutcomeError::HistoricalLabelArrival)
        );
        assert_eq!(
            monitor.record_outcome(5, 9, f64::NAN),
            Err(DelayedOutcomeError::NonFiniteOutcome)
        );

        monitor.record_outcome(5, 9, 1.0).unwrap();
        assert_eq!(
            monitor.record_outcome(5, 9, 2.0),
            Err(DelayedOutcomeError::DuplicateOutcome)
        );
    }

    #[test]
    fn alarm_must_be_current_and_precede_its_outcome() {
        let mut monitor = monitor(OutcomeDirection::HigherIsWorse);
        observe_through(&mut monitor, 2);
        assert_eq!(
            monitor.raise_alarm(0),
            Err(DelayedOutcomeError::AlarmNotAtCurrentStreamPosition)
        );
        monitor.record_outcome(1, 1, 2.0).unwrap();
        assert_eq!(
            monitor.raise_alarm(1),
            Err(DelayedOutcomeError::AlarmAfterOutcome)
        );
    }

    #[test]
    fn material_improvement_and_negligible_change_are_distinguished() {
        let mut stable = monitor(OutcomeDirection::HigherIsWorse);
        alarm_at_five(&mut stable);
        for index in 2..5 {
            stable.record_outcome(index, 9, 1.0).unwrap();
        }
        for index in 5..8 {
            stable.record_outcome(index, 9, 1.25).unwrap();
        }
        assert_eq!(
            stable.assess(5).unwrap().association,
            OutcomeAssociation::NoMaterialChange
        );

        let mut improved = monitor(OutcomeDirection::HigherIsWorse);
        alarm_at_five(&mut improved);
        for index in 2..5 {
            improved.record_outcome(index, 9, 3.0).unwrap();
        }
        for index in 5..8 {
            improved.record_outcome(index, 9, 1.0).unwrap();
        }
        let assessment = improved.assess(5).unwrap();
        assert_eq!(assessment.association, OutcomeAssociation::Improved);
        assert_eq!(assessment.worse_oriented_change, Some(-2.0));
    }

    #[test]
    fn numerical_overflow_never_becomes_a_no_change_result() {
        let mut monitor = monitor(OutcomeDirection::HigherIsWorse);
        alarm_at_five(&mut monitor);
        for index in 2..5 {
            monitor.record_outcome(index, 9, f64::MAX).unwrap();
        }
        for index in 5..8 {
            monitor.record_outcome(index, 9, f64::MAX).unwrap();
        }
        assert_eq!(
            monitor.assess(5).unwrap().association,
            OutcomeAssociation::NumericalOverflow
        );
    }

    #[test]
    fn invalid_configuration_is_rejected_before_stream_processing() {
        let valid = DelayedOutcomeConfig {
            window_size: 3,
            min_labeled_per_window: 2,
            min_labeled_fraction: 0.66,
            minimum_effect: 0.5,
            direction: OutcomeDirection::HigherIsWorse,
        };
        assert!(matches!(
            DelayedOutcomeMonitor::new(DelayedOutcomeConfig {
                window_size: 0,
                ..valid
            }),
            Err(DelayedOutcomeError::InvalidWindowSize)
        ));
        assert!(matches!(
            DelayedOutcomeMonitor::new(DelayedOutcomeConfig {
                min_labeled_per_window: 0,
                ..valid
            }),
            Err(DelayedOutcomeError::InvalidMinimumLabelCount)
        ));
        assert!(matches!(
            DelayedOutcomeMonitor::new(DelayedOutcomeConfig {
                min_labeled_fraction: f64::NAN,
                ..valid
            }),
            Err(DelayedOutcomeError::InvalidMinimumLabelFraction)
        ));
        assert!(matches!(
            DelayedOutcomeMonitor::new(DelayedOutcomeConfig {
                minimum_effect: f64::INFINITY,
                ..valid
            }),
            Err(DelayedOutcomeError::InvalidMinimumEffect)
        ));
    }

    #[test]
    fn stream_indices_and_alarms_fail_closed() {
        let mut monitor = monitor(OutcomeDirection::HigherIsWorse);
        monitor.observe(10).unwrap();
        assert_eq!(
            monitor.observe(12),
            Err(DelayedOutcomeError::NonSequentialObservation)
        );
        monitor.raise_alarm(10).unwrap();
        assert_eq!(
            monitor.raise_alarm(10),
            Err(DelayedOutcomeError::DuplicateAlarm)
        );
        assert_eq!(monitor.assess(9), Err(DelayedOutcomeError::UnknownAlarm));
    }
}
