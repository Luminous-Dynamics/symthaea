// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Threshold-feasibility classification over robust action-utility intervals.
//!
//! This module deliberately implements only the deterministic classification
//! layer. It does not calculate or verify confidence bounds, score receipts,
//! scenario support, model adequacy, operational mapping, or target transport.
//! Callers must supply the robust intervals from the registered inference path
//! and compose this result with all mandatory evidence gates before supporting
//! an operational decision.
//!
//! The family-level no-good-candidate result is returned only when every
//! supplied action has an upper bound strictly below the frozen threshold.
//! Budget exhaustion with any unresolved action is explicitly not a
//! no-good-candidate result.

use std::collections::BTreeSet;
use std::fmt;

use super::ranking::ActionRobustUtilityInterval;

/// Stable identity for this deterministic classification rule.
pub const ROBUST_THRESHOLD_CLASSIFICATION_METHOD_ID: &str =
    "spectra-robust-threshold-classification-v1";

/// Classification of one action against a predeclared minimum threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdActionStatus {
    /// The supplied robust lower bound is at least the required threshold.
    CertifiedGood,
    /// The supplied robust upper bound is strictly below the threshold.
    CertifiedBelowThreshold,
    /// The interval overlaps the threshold; neither conclusion is justified.
    Unresolved,
}

/// Per-action threshold result, retaining the exact bounds used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThresholdActionClassification {
    pub action_id: u32,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub status: ThresholdActionStatus,
}

/// Family-level conclusion over the complete supplied action set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdFamilyDisposition {
    /// At least one action is certified good. The complete good set is retained.
    GoodCandidateSetFound,
    /// Every action is certified below threshold; none meets the requirement.
    CertifiedNoGoodCandidate,
    /// No action is certified good, but at least one remains unresolved.
    /// This is the correct disposition when the available budget is exhausted.
    NotEstablishedAtBudget,
}

/// Deterministic threshold classification over one frozen set of action bounds.
///
/// Classifications and ID sets are returned in ascending action-ID order so
/// that output does not depend on input insertion order.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustThresholdClassification {
    pub method_id: &'static str,
    pub threshold: f64,
    pub action_classifications: Vec<ThresholdActionClassification>,
    pub certified_good_action_ids: Vec<u32>,
    pub certified_below_threshold_action_ids: Vec<u32>,
    pub unresolved_action_ids: Vec<u32>,
    pub disposition: ThresholdFamilyDisposition,
}

/// Invalid inputs to robust threshold classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustThresholdError {
    EmptyActionSet,
    InvalidThreshold,
    DuplicateActionId,
    InvalidUtilityInterval,
}

impl fmt::Display for RobustThresholdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyActionSet => "threshold classification requires at least one action",
            Self::InvalidThreshold => "threshold must be finite",
            Self::DuplicateActionId => "action IDs must be unique",
            Self::InvalidUtilityInterval => {
                "robust utility intervals must be finite and have lower <= upper"
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for RobustThresholdError {}

/// Classify a complete set of robust action intervals against a frozen threshold.
///
/// For each action a, with supplied robust interval [L(a), U(a)]:
///
/// - CertifiedGood iff L(a) >= threshold;
/// - CertifiedBelowThreshold iff U(a) < threshold;
/// - Unresolved otherwise.
///
/// The family disposition is CertifiedNoGoodCandidate only when every action
/// is CertifiedBelowThreshold. If any action is unresolved and none is
/// certified good, the result is NotEstablishedAtBudget.
///
/// This is a deterministic classifier, not an inference procedure or receipt
/// verifier. The caller is responsible for establishing that the intervals
/// are simultaneous, correctly scoped, and recomputed from a verified score
/// receipt. Mandatory model-adequacy, operational-mapping, target-support,
/// scenario-coverage, provenance, resource, and external-governance gates are
/// separate and may still block the composed decision.
pub fn classify_robust_threshold(
    action_intervals: &[ActionRobustUtilityInterval],
    threshold: f64,
) -> Result<RobustThresholdClassification, RobustThresholdError> {
    if action_intervals.is_empty() {
        return Err(RobustThresholdError::EmptyActionSet);
    }
    if !threshold.is_finite() {
        return Err(RobustThresholdError::InvalidThreshold);
    }

    let mut intervals = action_intervals.to_vec();
    intervals.sort_by_key(|interval| interval.action_id);

    let mut seen = BTreeSet::new();
    let mut action_classifications = Vec::with_capacity(intervals.len());
    let mut certified_good_action_ids = Vec::new();
    let mut certified_below_threshold_action_ids = Vec::new();
    let mut unresolved_action_ids = Vec::new();

    for interval in intervals {
        if !seen.insert(interval.action_id) {
            return Err(RobustThresholdError::DuplicateActionId);
        }
        if !interval.lower_bound.is_finite()
            || !interval.upper_bound.is_finite()
            || interval.lower_bound > interval.upper_bound
        {
            return Err(RobustThresholdError::InvalidUtilityInterval);
        }

        let status = if interval.lower_bound >= threshold {
            certified_good_action_ids.push(interval.action_id);
            ThresholdActionStatus::CertifiedGood
        } else if interval.upper_bound < threshold {
            certified_below_threshold_action_ids.push(interval.action_id);
            ThresholdActionStatus::CertifiedBelowThreshold
        } else {
            unresolved_action_ids.push(interval.action_id);
            ThresholdActionStatus::Unresolved
        };

        action_classifications.push(ThresholdActionClassification {
            action_id: interval.action_id,
            lower_bound: interval.lower_bound,
            upper_bound: interval.upper_bound,
            status,
        });
    }

    let disposition = if !certified_good_action_ids.is_empty() {
        ThresholdFamilyDisposition::GoodCandidateSetFound
    } else if unresolved_action_ids.is_empty() {
        ThresholdFamilyDisposition::CertifiedNoGoodCandidate
    } else {
        ThresholdFamilyDisposition::NotEstablishedAtBudget
    };

    Ok(RobustThresholdClassification {
        method_id: ROBUST_THRESHOLD_CLASSIFICATION_METHOD_ID,
        threshold,
        action_classifications,
        certified_good_action_ids,
        certified_below_threshold_action_ids,
        unresolved_action_ids,
        disposition,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(action_id: u32, lower_bound: f64, upper_bound: f64) -> ActionRobustUtilityInterval {
        ActionRobustUtilityInterval {
            action_id,
            lower_bound,
            upper_bound,
        }
    }

    #[test]
    fn distinguishes_good_below_and_unresolved_actions() {
        let result = classify_robust_threshold(
            &[
                interval(3, 0.72, 0.91),
                interval(1, 0.10, 0.35),
                interval(2, 0.42, 0.68),
            ],
            0.70,
        )
        .unwrap();

        assert_eq!(result.method_id, ROBUST_THRESHOLD_CLASSIFICATION_METHOD_ID);
        assert_eq!(result.disposition, ThresholdFamilyDisposition::GoodCandidateSetFound);
        assert_eq!(result.certified_good_action_ids, vec![3]);
        assert_eq!(result.certified_below_threshold_action_ids, vec![1]);
        assert_eq!(result.unresolved_action_ids, vec![2]);
        assert_eq!(
            result
                .action_classifications
                .iter()
                .map(|item| item.action_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn certifies_no_good_candidate_only_when_every_upper_bound_is_below_threshold() {
        let result = classify_robust_threshold(
            &[interval(9, 0.1, 0.4), interval(2, 0.3, 0.69)],
            0.70,
        )
        .unwrap();

        assert_eq!(result.disposition, ThresholdFamilyDisposition::CertifiedNoGoodCandidate);
        assert_eq!(result.certified_good_action_ids, Vec::<u32>::new());
        assert_eq!(result.certified_below_threshold_action_ids, vec![2, 9]);
        assert!(result.unresolved_action_ids.is_empty());
    }

    #[test]
    fn budget_exhaustion_with_unresolved_actions_is_not_a_no_good_result() {
        let result = classify_robust_threshold(
            &[interval(1, 0.2, 0.8), interval(2, 0.3, 0.6)],
            0.7,
        )
        .unwrap();

        assert_eq!(result.disposition, ThresholdFamilyDisposition::NotEstablishedAtBudget);
        assert!(result.certified_good_action_ids.is_empty());
        assert!(result.certified_below_threshold_action_ids.is_empty());
        assert_eq!(result.unresolved_action_ids, vec![1, 2]);
    }

    #[test]
    fn threshold_boundaries_are_conservative_and_unambiguous() {
        let result = classify_robust_threshold(
            &[
                interval(4, 0.7, 0.7),
                interval(3, 0.1, 0.7),
                interval(2, 0.7, 0.9),
                interval(1, 0.0, 0.699_999),
            ],
            0.7,
        )
        .unwrap();

        assert_eq!(result.certified_good_action_ids, vec![2, 4]);
        assert_eq!(result.certified_below_threshold_action_ids, vec![1]);
        assert_eq!(result.unresolved_action_ids, vec![3]);
        assert_eq!(result.disposition, ThresholdFamilyDisposition::GoodCandidateSetFound);
    }

    #[test]
    fn result_is_invariant_to_input_order() {
        let forward = [
            interval(1, 0.1, 0.2),
            interval(2, 0.8, 0.9),
            interval(3, 0.4, 0.7),
        ];
        let reverse = [forward[2], forward[1], forward[0]];

        assert_eq!(
            classify_robust_threshold(&forward, 0.75).unwrap(),
            classify_robust_threshold(&reverse, 0.75).unwrap()
        );
    }

    #[test]
    fn classifications_form_a_disjoint_complete_partition_on_deterministic_corpus() {
        let mut state = 0x5eed_u64;
        let mut next_unit = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / ((1_u64 << 53) as f64)
        };

        for _campaign in 0..256 {
            let threshold = 2.0 * next_unit() - 1.0;
            let mut intervals = Vec::with_capacity(32);
            for action_id in 0..32 {
                let first = 2.0 * next_unit() - 1.0;
                let second = 2.0 * next_unit() - 1.0;
                intervals.push(interval(
                    action_id,
                    first.min(second),
                    first.max(second),
                ));
            }

            let result = classify_robust_threshold(&intervals, threshold).unwrap();
            let partition_size = result.certified_good_action_ids.len()
                + result.certified_below_threshold_action_ids.len()
                + result.unresolved_action_ids.len();
            assert_eq!(partition_size, intervals.len());

            let mut partition_ids = result.certified_good_action_ids.clone();
            partition_ids.extend(result.certified_below_threshold_action_ids.iter().copied());
            partition_ids.extend(result.unresolved_action_ids.iter().copied());
            partition_ids.sort_unstable();
            partition_ids.dedup();
            assert_eq!(partition_ids.len(), intervals.len());
            assert_eq!(partition_ids, (0_u32..32).collect::<Vec<_>>());

            let expected_disposition = if !result.certified_good_action_ids.is_empty() {
                ThresholdFamilyDisposition::GoodCandidateSetFound
            } else if result.unresolved_action_ids.is_empty() {
                ThresholdFamilyDisposition::CertifiedNoGoodCandidate
            } else {
                ThresholdFamilyDisposition::NotEstablishedAtBudget
            };
            assert_eq!(result.disposition, expected_disposition);
        }
    }

    #[test]
    fn rejects_empty_duplicate_non_finite_and_inverted_inputs() {
        assert_eq!(
            classify_robust_threshold(&[], 0.5),
            Err(RobustThresholdError::EmptyActionSet)
        );
        assert_eq!(
            classify_robust_threshold(&[interval(1, 0.1, 0.9)], f64::NAN),
            Err(RobustThresholdError::InvalidThreshold)
        );
        assert_eq!(
            classify_robust_threshold(&[interval(1, 0.1, 0.9), interval(1, 0.2, 0.8)], 0.5),
            Err(RobustThresholdError::DuplicateActionId)
        );
        assert_eq!(
            classify_robust_threshold(&[interval(1, f64::NEG_INFINITY, 0.9)], 0.5),
            Err(RobustThresholdError::InvalidUtilityInterval)
        );
        assert_eq!(
            classify_robust_threshold(&[interval(1, 0.9, 0.1)], 0.5),
            Err(RobustThresholdError::InvalidUtilityInterval)
        );
    }
}
