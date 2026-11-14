// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Selective BCI evaluation: risk/coverage, calibration, and false activations.
//!
//! This module evaluates post-decoder policy thresholds using metadata-only labels.
//! Idle exposure must partition wall-clock time (not count overlapping windows more
//! than once) for false activations/hour to be interpretable.

use std::fmt;

const SCALE: u128 = 1000;
const NS_PER_HOUR: u128 = 3_600_000_000_000;
const ECE_BINS: usize = 10;

/// One labelled decision interval. prediction_correct means the decoder's top
/// predicted class matches ground truth, including the explicit idle class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectiveObservation {
    /// If false, the independent signal-quality gate blocks all actions.
    pub quality_eligible: bool,
    /// Top-class decoder confidence in [0, 1000].
    pub confidence_permille: u16,
    pub prediction_correct: bool,
    /// True only on the onset of a newly emitted non-idle command, not on each
    /// interval while a command remains active. This prevents one long hold from
    /// being counted as many false-activation events.
    pub action_onset: bool,
    /// Ground truth for this interval is idle/no command.
    pub target_is_idle: bool,
    /// Duration represented by this interval; idle intervals must be disjoint for
    /// a true wall-clock exposure denominator.
    pub interval_duration_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectiveMetricError {
    EmptyObservations,
    EmptyThresholds,
    InvalidThreshold(u16),
    DuplicateThreshold(u16),
    InvalidConfidence { index: usize, value: u16 },
    CorrectPredictionWithIdleActionOnset { index: usize },
    CounterOverflow,
}

impl fmt::Display for SelectiveMetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SelectiveMetricError {}

/// Results are integer-scaled for deterministic serialization.
/// Permille values are in [0, 1000]. ECE uses ten fixed-width confidence bins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectiveMetrics {
    pub threshold_permille: u16,
    pub total_observations: u64,
    pub quality_eligible_observations: u64,
    pub accepted_observations: u64,
    pub accepted_correct: u64,
    pub coverage_permille: u16,
    /// Error rate among accepted decisions. None means no decisions were accepted.
    pub selective_risk_permille: Option<u16>,
    /// Expected calibration error among accepted decisions; None when none accepted.
    pub ece_permille: Option<u16>,
    pub false_activation_events: u64,
    pub eligible_idle_exposure_ns: u128,
    /// False activations/hour scaled by 1000; None if no eligible idle exposure exists.
    pub false_activations_per_hour_milli: Option<u128>,
}

pub fn evaluate_selective(
    observations: &[SelectiveObservation],
    threshold_permille: u16,
) -> Result<SelectiveMetrics, SelectiveMetricError> {
    if observations.is_empty() {
        return Err(SelectiveMetricError::EmptyObservations);
    }
    if threshold_permille > 1000 {
        return Err(SelectiveMetricError::InvalidThreshold(threshold_permille));
    }

    let total = u64::try_from(observations.len()).map_err(|_| SelectiveMetricError::CounterOverflow)?;
    let mut eligible = 0_u64;
    let mut accepted = 0_u64;
    let mut correct = 0_u64;
    let mut false_activations = 0_u64;
    let mut idle_exposure_ns = 0_u128;
    let mut bin_confidence_sum = [0_u128; ECE_BINS];
    let mut bin_correct = [0_u128; ECE_BINS];

    for (index, item) in observations.iter().enumerate() {
        if item.confidence_permille > 1000 {
            return Err(SelectiveMetricError::InvalidConfidence {
                index,
                value: item.confidence_permille,
            });
        }
        if item.target_is_idle && item.action_onset && item.prediction_correct {
            return Err(SelectiveMetricError::CorrectPredictionWithIdleActionOnset { index });
        }

        if item.quality_eligible {
            eligible = eligible.checked_add(1).ok_or(SelectiveMetricError::CounterOverflow)?;
            if item.target_is_idle {
                idle_exposure_ns = idle_exposure_ns
                    .checked_add(u128::from(item.interval_duration_ns))
                    .ok_or(SelectiveMetricError::CounterOverflow)?;
            }
        }

        let accepted_here = item.quality_eligible && item.confidence_permille >= threshold_permille;
        if !accepted_here {
            continue;
        }

        accepted = accepted.checked_add(1).ok_or(SelectiveMetricError::CounterOverflow)?;
        if item.prediction_correct {
            correct = correct.checked_add(1).ok_or(SelectiveMetricError::CounterOverflow)?;
        }
        if item.target_is_idle && item.action_onset {
            false_activations = false_activations.checked_add(1)
                .ok_or(SelectiveMetricError::CounterOverflow)?;
        }

        let bin = usize::from(item.confidence_permille / 100).min(ECE_BINS - 1);
        bin_confidence_sum[bin] += u128::from(item.confidence_permille);
        if item.prediction_correct {
            bin_correct[bin] += 1;
        }
    }

    let coverage_permille = u16::try_from((u128::from(accepted) * SCALE) / u128::from(total))
        .map_err(|_| SelectiveMetricError::CounterOverflow)?;
    let selective_risk_permille = if accepted == 0 {
        None
    } else {
        Some(u16::try_from((u128::from(accepted - correct) * SCALE) / u128::from(accepted))
            .map_err(|_| SelectiveMetricError::CounterOverflow)?)
    };

    // Weighted fixed-bin ECE in permille:
    // sum_b |sum(confidence_b) - 1000 * correct_b| / total_accepted.
    let mut calibration_deviation = 0_u128;
    for (confidence_sum, correct_count) in bin_confidence_sum.iter().zip(bin_correct.iter()) {
        let expected = *correct_count * SCALE;
        calibration_deviation = calibration_deviation
            .checked_add(confidence_sum.abs_diff(expected))
            .ok_or(SelectiveMetricError::CounterOverflow)?;
    }
    let ece_permille = if accepted == 0 {
        None
    } else {
        Some(u16::try_from(calibration_deviation / u128::from(accepted))
            .map_err(|_| SelectiveMetricError::CounterOverflow)?)
    };

    let false_activations_per_hour_milli = if idle_exposure_ns == 0 {
        None
    } else {
        Some((u128::from(false_activations) * NS_PER_HOUR * SCALE) / idle_exposure_ns)
    };

    Ok(SelectiveMetrics {
        threshold_permille,
        total_observations: total,
        quality_eligible_observations: eligible,
        accepted_observations: accepted,
        accepted_correct: correct,
        coverage_permille,
        selective_risk_permille,
        ece_permille,
        false_activation_events: false_activations,
        eligible_idle_exposure_ns: idle_exposure_ns,
        false_activations_per_hour_milli,
    })
}

/// Evaluate a preregistered set of thresholds to obtain a risk/coverage curve.
/// Threshold order is preserved; duplicate thresholds are rejected.
pub fn evaluate_risk_coverage(
    observations: &[SelectiveObservation],
    thresholds_permille: &[u16],
) -> Result<Vec<SelectiveMetrics>, SelectiveMetricError> {
    if thresholds_permille.is_empty() {
        return Err(SelectiveMetricError::EmptyThresholds);
    }
    let mut seen = std::collections::BTreeSet::new();
    for &threshold in thresholds_permille {
        if threshold > 1000 {
            return Err(SelectiveMetricError::InvalidThreshold(threshold));
        }
        if !seen.insert(threshold) {
            return Err(SelectiveMetricError::DuplicateThreshold(threshold));
        }
    }
    thresholds_permille.iter()
        .map(|threshold| evaluate_selective(observations, *threshold))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(confidence: u16, eligible: bool, correct: bool, action_onset: bool, idle: bool, duration_ns: u64) -> SelectiveObservation {
        SelectiveObservation {
            quality_eligible: eligible,
            confidence_permille: confidence,
            prediction_correct: correct,
            action_onset,
            target_is_idle: idle,
            interval_duration_ns: duration_ns,
        }
    }

    #[test]
    fn reports_coverage_risk_and_false_activation_rate() {
        let one_hour = 3_600_000_000_000_u64;
        let data = [
            observation(950, true, true, true, false, 0),
            observation(900, true, false, true, true, one_hour / 2),
            observation(400, true, true, false, true, one_hour / 2),
            observation(990, false, false, true, true, one_hour),
        ];
        let m = evaluate_selective(&data, 800).unwrap();
        assert_eq!(m.total_observations, 4);
        assert_eq!(m.quality_eligible_observations, 3);
        assert_eq!(m.accepted_observations, 2);
        assert_eq!(m.accepted_correct, 1);
        assert_eq!(m.coverage_permille, 500);
        assert_eq!(m.selective_risk_permille, Some(500));
        assert_eq!(m.false_activation_events, 1);
        assert_eq!(m.false_activations_per_hour_milli, Some(1000));
    }

    #[test]
    fn abstains_when_quality_gate_blocks_all_observations() {
        let data = [observation(1000, false, true, false, true, 3_600_000_000_000)];
        let m = evaluate_selective(&data, 0).unwrap();
        assert_eq!(m.coverage_permille, 0);
        assert_eq!(m.selective_risk_permille, None);
        assert_eq!(m.ece_permille, None);
        assert_eq!(m.false_activations_per_hour_milli, None);
    }

    #[test]
    fn threshold_sweep_monotonically_reduces_coverage() {
        let data = [
            observation(200, true, true, false, true, 100),
            observation(600, true, false, true, true, 100),
            observation(900, true, true, true, false, 0),
        ];
        let result = evaluate_risk_coverage(&data, &[0, 500, 800, 1000]).unwrap();
        assert_eq!(result.iter().map(|m| m.coverage_permille).collect::<Vec<_>>(), vec![1000, 666, 333, 0]);
    }

    #[test]
    fn rejects_empty_inputs_duplicate_thresholds_and_invalid_confidence() {
        assert_eq!(evaluate_selective(&[], 500), Err(SelectiveMetricError::EmptyObservations));
        assert_eq!(evaluate_risk_coverage(&[], &[]), Err(SelectiveMetricError::EmptyThresholds));
        let data = [observation(1001, true, true, false, false, 0)];
        assert_eq!(evaluate_selective(&data, 500), Err(SelectiveMetricError::InvalidConfidence { index: 0, value: 1001 }));
        let data = [observation(500, true, true, false, false, 0)];
        assert_eq!(evaluate_risk_coverage(&data, &[500, 500]), Err(SelectiveMetricError::DuplicateThreshold(500)));
    }

    #[test]
    fn rejects_ground_truth_inconsistent_with_false_action_onset() {
        let data = [observation(900, true, true, true, true, 1_000)];
        assert_eq!(
            evaluate_selective(&data, 500),
            Err(SelectiveMetricError::CorrectPredictionWithIdleActionOnset { index: 0 })
        );
    }

    #[test]
    fn confidence_calibration_error_is_zero_for_perfect_predictions() {
        let data = [
            observation(1000, true, true, false, true, 10),
            observation(0, true, false, false, true, 10),
        ];
        let m = evaluate_selective(&data, 0).unwrap();
        assert_eq!(m.ece_permille, Some(0));
    }
}
