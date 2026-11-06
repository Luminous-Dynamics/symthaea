// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conjunctive multi-constraint feasibility classification.
//!
//! This module performs deterministic classification over a complete frozen
//! action × constraint interval grid. It does not estimate or authenticate the
//! intervals, establish simultaneous coverage, or qualify external evidence
//! gates. Unknown or unresolved constraints are never silently treated as pass.

use std::collections::BTreeSet;
use std::fmt;

/// Stable identity for the conjunctive multi-threshold decision rule.
pub const MULTI_THRESHOLD_CLASSIFICATION_METHOD_ID: &str =
    "spectra-conjunctive-multi-threshold-classification-v1";

/// Direction of a hard constraint after choosing its natural comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintDirection {
    /// A value at least the threshold is acceptable.
    AtLeast,
    /// A value at most the threshold is acceptable.
    AtMost,
}

/// Frozen threshold for one constraint dimension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstraintThreshold {
    pub constraint_id: u16,
    pub direction: ConstraintDirection,
    pub threshold: f64,
}

/// Robust interval for one action and one constraint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionConstraintInterval {
    pub action_id: u32,
    pub constraint_id: u16,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

/// Frozen complete decision universe.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiThresholdContract {
    pub action_ids: Vec<u32>,
    pub constraints: Vec<ConstraintThreshold>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintDisposition {
    CertifiedPass,
    CertifiedViolation,
    Unresolved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionFeasibilityDisposition {
    /// Every frozen constraint is certified satisfied.
    CertifiedFeasible,
    /// At least one frozen constraint is definitely violated.
    CertifiedInfeasible,
    /// No definite violation exists, but at least one constraint is unresolved.
    Unresolved,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstraintClassification {
    pub constraint_id: u16,
    pub direction: ConstraintDirection,
    pub threshold: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub disposition: ConstraintDisposition,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionMultiThresholdClassification {
    pub action_id: u32,
    pub disposition: ActionFeasibilityDisposition,
    pub constraint_classifications: Vec<ConstraintClassification>,
    pub violation_constraint_ids: Vec<u16>,
    pub unresolved_constraint_ids: Vec<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiThresholdFamilyDisposition {
    /// The complete set of actions meeting every hard constraint is nonempty.
    GoodCandidateSetFound,
    /// Every frozen action is definitely infeasible.
    CertifiedNoGoodCandidate,
    /// No good action is found, but at least one action remains unresolved.
    NotEstablishedAtBudget,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiThresholdClassification {
    pub method_id: &'static str,
    pub contract: MultiThresholdContract,
    pub action_classifications: Vec<ActionMultiThresholdClassification>,
    pub certified_good_action_ids: Vec<u32>,
    pub certified_infeasible_action_ids: Vec<u32>,
    pub unresolved_action_ids: Vec<u32>,
    pub disposition: MultiThresholdFamilyDisposition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiThresholdError {
    EmptyActionSet,
    EmptyConstraintSet,
    DuplicateActionId,
    DuplicateConstraintId,
    InvalidThreshold,
    IncompleteIntervalGrid,
    DuplicateActionConstraintInterval,
    UnknownActionId,
    UnknownConstraintId,
    InvalidInterval,
    IntervalCountOverflow,
}

impl fmt::Display for MultiThresholdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyActionSet => "multi-threshold classification requires a nonempty frozen action set",
            Self::EmptyConstraintSet => "multi-threshold classification requires at least one frozen constraint",
            Self::DuplicateActionId => "frozen action IDs must be unique",
            Self::DuplicateConstraintId => "frozen constraint IDs must be unique",
            Self::InvalidThreshold => "every frozen constraint threshold must be finite",
            Self::IncompleteIntervalGrid => "exactly one interval is required for every action × constraint pair",
            Self::DuplicateActionConstraintInterval => "duplicate interval for an action × constraint pair",
            Self::UnknownActionId => "interval references an action outside the frozen action set",
            Self::UnknownConstraintId => "interval references a constraint outside the frozen constraint set",
            Self::InvalidInterval => "constraint interval must be finite and ordered",
            Self::IntervalCountOverflow => "action × constraint grid size overflowed usize",
        };
        f.write_str(message)
    }
}
impl std::error::Error for MultiThresholdError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiThresholdReplayError {
    InvalidInput(MultiThresholdError),
    ClassificationMismatch,
}

impl fmt::Display for MultiThresholdReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(error) => write!(f, "cannot replay multi-threshold classification: {error}"),
            Self::ClassificationMismatch => {
                f.write_str("recorded multi-threshold classification does not match deterministic recomputation")
            }
        }
    }
}
impl std::error::Error for MultiThresholdReplayError {}
impl From<MultiThresholdError> for MultiThresholdReplayError {
    fn from(error: MultiThresholdError) -> Self {
        Self::InvalidInput(error)
    }
}

/// Classify every action against every frozen hard constraint.
///
/// For AtLeast(τ): pass iff L ≥ τ; definite violation iff U < τ.
/// For AtMost(τ): pass iff U ≤ τ; definite violation iff L > τ.
/// All remaining cases are unresolved.
///
/// An action is certified feasible iff every constraint passes. It is
/// certified infeasible if any one constraint is definitely violated, while
/// all other per-constraint outcomes remain visible. The family-level
/// no-good result is returned only if every action is definitely infeasible.
///
/// The inferential meaning is conditional on the upstream action × constraint
/// intervals being jointly valid over the complete frozen grid. Since the
/// classification is deterministic post-processing of that grid, no second
/// union bound over the action family is needed. Mandatory provenance,
/// support, model-adequacy and governance gates remain separate.
pub fn classify_multi_threshold_constraints(
    intervals: &[ActionConstraintInterval],
    contract: &MultiThresholdContract,
) -> Result<MultiThresholdClassification, MultiThresholdError> {
    if contract.action_ids.is_empty() {
        return Err(MultiThresholdError::EmptyActionSet);
    }
    if contract.constraints.is_empty() {
        return Err(MultiThresholdError::EmptyConstraintSet);
    }

    let mut action_ids = contract.action_ids.clone();
    action_ids.sort_unstable();
    if action_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(MultiThresholdError::DuplicateActionId);
    }

    let mut constraints = contract.constraints.clone();
    constraints.sort_by_key(|constraint| constraint.constraint_id);
    if constraints.windows(2).any(|pair| pair[0].constraint_id == pair[1].constraint_id) {
        return Err(MultiThresholdError::DuplicateConstraintId);
    }
    if constraints.iter().any(|constraint| !constraint.threshold.is_finite()) {
        return Err(MultiThresholdError::InvalidThreshold);
    }

    let expected_count = action_ids
        .len()
        .checked_mul(constraints.len())
        .ok_or(MultiThresholdError::IntervalCountOverflow)?;
    if intervals.len() != expected_count {
        return Err(MultiThresholdError::IncompleteIntervalGrid);
    }

    let known_actions: BTreeSet<u32> = action_ids.iter().copied().collect();
    let known_constraints: BTreeSet<u16> =
        constraints.iter().map(|constraint| constraint.constraint_id).collect();
    let mut by_pair = std::collections::BTreeMap::new();

    for interval in intervals {
        if !known_actions.contains(&interval.action_id) {
            return Err(MultiThresholdError::UnknownActionId);
        }
        if !known_constraints.contains(&interval.constraint_id) {
            return Err(MultiThresholdError::UnknownConstraintId);
        }
        if !interval.lower_bound.is_finite()
            || !interval.upper_bound.is_finite()
            || interval.lower_bound > interval.upper_bound
        {
            return Err(MultiThresholdError::InvalidInterval);
        }
        if by_pair
            .insert((interval.action_id, interval.constraint_id), *interval)
            .is_some()
        {
            return Err(MultiThresholdError::DuplicateActionConstraintInterval);
        }
    }

    if by_pair.len() != expected_count {
        return Err(MultiThresholdError::IncompleteIntervalGrid);
    }

    let mut action_classifications = Vec::with_capacity(action_ids.len());
    let mut certified_good_action_ids = Vec::new();
    let mut certified_infeasible_action_ids = Vec::new();
    let mut unresolved_action_ids = Vec::new();

    for action_id in action_ids {
        let mut constraint_classifications = Vec::with_capacity(constraints.len());
        let mut violation_constraint_ids = Vec::new();
        let mut unresolved_constraint_ids = Vec::new();

        for constraint in &constraints {
            let interval = by_pair
                .get(&(action_id, constraint.constraint_id))
                .copied()
                .ok_or(MultiThresholdError::IncompleteIntervalGrid)?;
            let disposition = match constraint.direction {
                ConstraintDirection::AtLeast if interval.lower_bound >= constraint.threshold => {
                    ConstraintDisposition::CertifiedPass
                }
                ConstraintDirection::AtLeast if interval.upper_bound < constraint.threshold => {
                    ConstraintDisposition::CertifiedViolation
                }
                ConstraintDirection::AtMost if interval.upper_bound <= constraint.threshold => {
                    ConstraintDisposition::CertifiedPass
                }
                ConstraintDirection::AtMost if interval.lower_bound > constraint.threshold => {
                    ConstraintDisposition::CertifiedViolation
                }
                _ => ConstraintDisposition::Unresolved,
            };

            match disposition {
                ConstraintDisposition::CertifiedPass => {}
                ConstraintDisposition::CertifiedViolation => {
                    violation_constraint_ids.push(constraint.constraint_id);
                }
                ConstraintDisposition::Unresolved => {
                    unresolved_constraint_ids.push(constraint.constraint_id);
                }
            }
            constraint_classifications.push(ConstraintClassification {
                constraint_id: constraint.constraint_id,
                direction: constraint.direction,
                threshold: constraint.threshold,
                lower_bound: interval.lower_bound,
                upper_bound: interval.upper_bound,
                disposition,
            });
        }

        let disposition = if !violation_constraint_ids.is_empty() {
            certified_infeasible_action_ids.push(action_id);
            ActionFeasibilityDisposition::CertifiedInfeasible
        } else if !unresolved_constraint_ids.is_empty() {
            unresolved_action_ids.push(action_id);
            ActionFeasibilityDisposition::Unresolved
        } else {
            certified_good_action_ids.push(action_id);
            ActionFeasibilityDisposition::CertifiedFeasible
        };

        action_classifications.push(ActionMultiThresholdClassification {
            action_id,
            disposition,
            constraint_classifications,
            violation_constraint_ids,
            unresolved_constraint_ids,
        });
    }

    let disposition = if !certified_good_action_ids.is_empty() {
        MultiThresholdFamilyDisposition::GoodCandidateSetFound
    } else if certified_infeasible_action_ids.len() == action_classifications.len() {
        MultiThresholdFamilyDisposition::CertifiedNoGoodCandidate
    } else {
        MultiThresholdFamilyDisposition::NotEstablishedAtBudget
    };

    let mut canonical_contract = contract.clone();
    canonical_contract.action_ids = action_classifications.iter().map(|action| action.action_id).collect();
    canonical_contract.constraints = constraints;

    Ok(MultiThresholdClassification {
        method_id: MULTI_THRESHOLD_CLASSIFICATION_METHOD_ID,
        contract: canonical_contract,
        action_classifications,
        certified_good_action_ids,
        certified_infeasible_action_ids,
        unresolved_action_ids,
        disposition,
    })
}

/// Replay every classification and witness against the frozen input grid.
pub fn verify_multi_threshold_classification_replay(
    intervals: &[ActionConstraintInterval],
    contract: &MultiThresholdContract,
    recorded: &MultiThresholdClassification,
) -> Result<(), MultiThresholdReplayError> {
    let expected = classify_multi_threshold_constraints(intervals, contract)?;
    if &expected == recorded {
        Ok(())
    } else {
        Err(MultiThresholdReplayError::ClassificationMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> MultiThresholdContract {
        MultiThresholdContract {
            action_ids: vec![1, 2, 3],
            constraints: vec![
                ConstraintThreshold {
                    constraint_id: 10,
                    direction: ConstraintDirection::AtLeast,
                    threshold: 0.7,
                },
                ConstraintThreshold {
                    constraint_id: 20,
                    direction: ConstraintDirection::AtMost,
                    threshold: 0.3,
                },
            ],
        }
    }

    fn interval(action_id: u32, constraint_id: u16, lower: f64, upper: f64) -> ActionConstraintInterval {
        ActionConstraintInterval {
            action_id,
            constraint_id,
            lower_bound: lower,
            upper_bound: upper,
        }
    }

    #[test]
    fn finds_complete_good_set_and_preserves_other_constraint_witnesses() {
        let intervals = [
            interval(1, 10, 0.8, 0.9),
            interval(1, 20, 0.1, 0.2),
            interval(2, 10, 0.2, 0.3),
            interval(2, 20, 0.2, 0.4),
            interval(3, 10, 0.6, 0.8),
            interval(3, 20, 0.4, 0.5),
        ];
        let result = classify_multi_threshold_constraints(&intervals, &contract()).unwrap();

        assert_eq!(result.certified_good_action_ids, vec![1]);
        assert_eq!(result.certified_infeasible_action_ids, vec![2, 3]);
        assert!(result.unresolved_action_ids.is_empty());
        assert_eq!(result.disposition, MultiThresholdFamilyDisposition::GoodCandidateSetFound);
        assert_eq!(result.action_classifications[1].violation_constraint_ids, vec![10]);
        assert_eq!(result.action_classifications[1].unresolved_constraint_ids, vec![20]);
        assert_eq!(result.action_classifications[2].violation_constraint_ids, vec![20]);
        assert_eq!(result.action_classifications[2].unresolved_constraint_ids, vec![10]);
    }

    #[test]
    fn certifies_no_good_only_when_every_action_has_a_definite_violation() {
        let intervals = [
            interval(1, 10, 0.1, 0.2),
            interval(1, 20, 0.1, 0.2),
            interval(2, 10, 0.8, 0.9),
            interval(2, 20, 0.4, 0.5),
            interval(3, 10, 0.2, 0.3),
            interval(3, 20, 0.4, 0.5),
        ];
        let result = classify_multi_threshold_constraints(&intervals, &contract()).unwrap();
        assert!(result.certified_good_action_ids.is_empty());
        assert_eq!(result.certified_infeasible_action_ids, vec![1, 2, 3]);
        assert!(result.unresolved_action_ids.is_empty());
        assert_eq!(result.disposition, MultiThresholdFamilyDisposition::CertifiedNoGoodCandidate);
    }

    #[test]
    fn unresolved_action_prevents_no_good_conclusion_when_no_action_is_good() {
        let intervals = [
            interval(1, 10, 0.1, 0.2),
            interval(1, 20, 0.1, 0.2),
            interval(2, 10, 0.6, 0.8),
            interval(2, 20, 0.1, 0.2),
            interval(3, 10, 0.2, 0.3),
            interval(3, 20, 0.4, 0.5),
        ];
        let result = classify_multi_threshold_constraints(&intervals, &contract()).unwrap();
        assert_eq!(result.certified_infeasible_action_ids, vec![1, 3]);
        assert_eq!(result.unresolved_action_ids, vec![2]);
        assert_eq!(result.disposition, MultiThresholdFamilyDisposition::NotEstablishedAtBudget);
    }

    #[test]
    fn both_at_least_and_at_most_boundaries_are_conservative() {
        let contract = MultiThresholdContract {
            action_ids: vec![1, 2, 3, 4],
            constraints: vec![
                ConstraintThreshold { constraint_id: 1, direction: ConstraintDirection::AtLeast, threshold: 0.7 },
                ConstraintThreshold { constraint_id: 2, direction: ConstraintDirection::AtMost, threshold: 0.3 },
            ],
        };
        let intervals = [
            interval(1, 1, 0.7, 0.8), // AtLeast passes at lower == threshold
            interval(1, 2, 0.1, 0.3), // AtMost passes at upper == threshold
            interval(2, 1, 0.2, 0.7), // AtLeast unresolved at upper == threshold
            interval(2, 2, 0.1, 0.3),
            interval(3, 1, 0.7, 0.9),
            interval(3, 2, 0.3, 0.5), // AtMost unresolved at lower == threshold
            interval(4, 1, 0.1, 0.69), // AtLeast definite violation
            interval(4, 2, 0.1, 0.2),
        ];
        let result = classify_multi_threshold_constraints(&intervals, &contract).unwrap();
        assert_eq!(result.certified_good_action_ids, vec![1]);
        assert_eq!(result.unresolved_action_ids, vec![2, 3]);
        assert_eq!(result.certified_infeasible_action_ids, vec![4]);
        assert_eq!(result.disposition, MultiThresholdFamilyDisposition::GoodCandidateSetFound);
    }

    #[test]
    fn output_is_invariant_to_action_constraint_and_interval_order() {
        let original_contract = contract();
        let mut shuffled_contract = contract();
        shuffled_contract.action_ids.reverse();
        shuffled_contract.constraints.reverse();
        let mut intervals = vec![
            interval(1, 10, 0.8, 0.9), interval(1, 20, 0.1, 0.2),
            interval(2, 10, 0.2, 0.3), interval(2, 20, 0.2, 0.4),
            interval(3, 10, 0.6, 0.8), interval(3, 20, 0.4, 0.5),
        ];
        intervals.reverse();
        assert_eq!(
            classify_multi_threshold_constraints(&intervals, &shuffled_contract).unwrap(),
            classify_multi_threshold_constraints(&[
                interval(1, 10, 0.8, 0.9), interval(1, 20, 0.1, 0.2),
                interval(2, 10, 0.2, 0.3), interval(2, 20, 0.2, 0.4),
                interval(3, 10, 0.6, 0.8), interval(3, 20, 0.4, 0.5),
            ], &original_contract).unwrap()
        );
    }

    #[test]
    fn replay_accepts_exact_output_and_rejects_mutated_witness_or_disposition() {
        let intervals = [
            interval(1, 10, 0.8, 0.9), interval(1, 20, 0.1, 0.2),
            interval(2, 10, 0.2, 0.3), interval(2, 20, 0.2, 0.4),
            interval(3, 10, 0.6, 0.8), interval(3, 20, 0.4, 0.5),
        ];
        let c = contract();
        let expected = classify_multi_threshold_constraints(&intervals, &c).unwrap();
        assert_eq!(verify_multi_threshold_classification_replay(&intervals, &c, &expected), Ok(()));

        let mut changed_witness = expected.clone();
        changed_witness.action_classifications[1].violation_constraint_ids.clear();
        assert_eq!(
            verify_multi_threshold_classification_replay(&intervals, &c, &changed_witness),
            Err(MultiThresholdReplayError::ClassificationMismatch)
        );

        let mut changed_disposition = expected.clone();
        changed_disposition.disposition = MultiThresholdFamilyDisposition::CertifiedNoGoodCandidate;
        assert_eq!(
            verify_multi_threshold_classification_replay(&intervals, &c, &changed_disposition),
            Err(MultiThresholdReplayError::ClassificationMismatch)
        );
    }

    #[test]
    fn rejects_incomplete_duplicate_unknown_and_invalid_grids() {
        let c = contract();
        let mut incomplete = vec![
            interval(1, 10, 0.8, 0.9), interval(1, 20, 0.1, 0.2),
            interval(2, 10, 0.2, 0.3), interval(2, 20, 0.2, 0.4),
            interval(3, 10, 0.6, 0.8), interval(3, 20, 0.4, 0.5),
        ];
        incomplete.pop();
        assert_eq!(classify_multi_threshold_constraints(&incomplete, &c), Err(MultiThresholdError::IncompleteIntervalGrid));

        let mut duplicate = vec![
            interval(1, 10, 0.8, 0.9), interval(1, 20, 0.1, 0.2),
            interval(2, 10, 0.2, 0.3), interval(2, 20, 0.2, 0.4),
            interval(3, 10, 0.6, 0.8), interval(3, 20, 0.4, 0.5),
        ];
        duplicate[1] = duplicate[0];
        assert_eq!(classify_multi_threshold_constraints(&duplicate, &c), Err(MultiThresholdError::DuplicateActionConstraintInterval));

        let unknown_action = [
            interval(99, 10, 0.8, 0.9),
            interval(1, 20, 0.1, 0.2),
            interval(2, 10, 0.2, 0.3),
            interval(2, 20, 0.2, 0.4),
            interval(3, 10, 0.6, 0.8),
            interval(3, 20, 0.4, 0.5),
        ];
        assert_eq!(
            classify_multi_threshold_constraints(&unknown_action, &c),
            Err(MultiThresholdError::UnknownActionId)
        );
        let unknown_constraint = [
            interval(1, 99, 0.8, 0.9),
            interval(1, 20, 0.1, 0.2),
            interval(2, 10, 0.2, 0.3),
            interval(2, 20, 0.2, 0.4),
            interval(3, 10, 0.6, 0.8),
            interval(3, 20, 0.4, 0.5),
        ];
        assert_eq!(
            classify_multi_threshold_constraints(&unknown_constraint, &c),
            Err(MultiThresholdError::UnknownConstraintId)
        );
        assert_eq!(
            classify_multi_threshold_constraints(&[], &MultiThresholdContract {
                action_ids: vec![],
                constraints: c.constraints.clone(),
            }),
            Err(MultiThresholdError::EmptyActionSet)
        );
        assert_eq!(
            classify_multi_threshold_constraints(&[], &MultiThresholdContract {
                action_ids: vec![1],
                constraints: vec![],
            }),
            Err(MultiThresholdError::EmptyConstraintSet)
        );

        let mut bad_threshold = c.clone();
        bad_threshold.constraints[0].threshold = f64::NAN;
        let full = vec![
            interval(1, 10, 0.8, 0.9), interval(1, 20, 0.1, 0.2),
            interval(2, 10, 0.2, 0.3), interval(2, 20, 0.2, 0.4),
            interval(3, 10, 0.6, 0.8), interval(3, 20, 0.4, 0.5),
        ];
        assert_eq!(classify_multi_threshold_constraints(&full, &bad_threshold), Err(MultiThresholdError::InvalidThreshold));
    }

    #[test]
    fn deterministic_corpus_always_partitions_every_action_exactly_once() {
        let mut state = 0x5eed_4d54_u64;
        let mut next_unit = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / ((1_u64 << 53) as f64)
        };
        for _campaign in 0..256 {
            let mut c = MultiThresholdContract {
                action_ids: (0..32).collect(),
                constraints: vec![
                    ConstraintThreshold { constraint_id: 1, direction: ConstraintDirection::AtLeast, threshold: 2.0 * next_unit() - 1.0 },
                    ConstraintThreshold { constraint_id: 2, direction: ConstraintDirection::AtMost, threshold: 2.0 * next_unit() - 1.0 },
                ],
            };
            let mut intervals = Vec::new();
            for action_id in &c.action_ids {
                for constraint in &c.constraints {
                    let a = 2.0 * next_unit() - 1.0;
                    let b = 2.0 * next_unit() - 1.0;
                    intervals.push(interval(*action_id, constraint.constraint_id, a.min(b), a.max(b)));
                }
            }
            let result = classify_multi_threshold_constraints(&intervals, &c).unwrap();
            let total = result.certified_good_action_ids.len()
                + result.certified_infeasible_action_ids.len()
                + result.unresolved_action_ids.len();
            assert_eq!(total, 32);
            let mut all = result.certified_good_action_ids.clone();
            all.extend(result.certified_infeasible_action_ids.iter().copied());
            all.extend(result.unresolved_action_ids.iter().copied());
            all.sort_unstable();
            all.dedup();
            assert_eq!(all, (0_u32..32).collect::<Vec<_>>());
            c.action_ids.reverse();
        }
    }
}
