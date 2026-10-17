// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conservative Pareto-front extraction when objective values have uncertainty bounds.
//!
//! This module does not estimate uncertainty. The caller must supply bounds derived
//! from a versioned measurement or prediction procedure, with matching semantics
//! across candidates. Overlapping intervals remain unresolved rather than being
//! converted into a point-estimate win. Hard constraints and evidence eligibility
//! must be enforced before calling this module.

use std::collections::HashSet;
use std::fmt;

use crate::pareto::{ObjectiveDirection, ObjectiveSpec};

/// A finite interval bounding one objective value under a declared uncertainty profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectiveInterval {
    pub lower: f64,
    pub upper: f64,
}

impl ObjectiveInterval {
    /// Build an interval. Ordering and finiteness are checked before ranking.
    pub fn new(lower: f64, upper: f64) -> Self {
        Self { lower, upper }
    }

    /// Construct a point-valued objective with no represented uncertainty.
    ///
    /// This is useful for compatibility checks against point-estimate Pareto
    /// ranking. It must not be used to imply that physical uncertainty is zero.
    pub fn point(value: f64) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }
}

/// A candidate evaluated under one exact, caller-defined uncertainty profile.
///
/// bounds must follow the same order as the objective specifications. The
/// profile identifier is an opaque versioned contract: it should pin units,
/// interval semantics, analysis procedure and relevant conditions, not merely
/// a human-readable label.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedCandidate {
    pub id: String,
    pub uncertainty_profile_id: String,
    pub bounds: Vec<ObjectiveInterval>,
}

impl BoundedCandidate {
    /// Construct a bounded candidate with a stable identifier and profile ID.
    pub fn new(
        id: impl Into<String>,
        uncertainty_profile_id: impl Into<String>,
        bounds: Vec<ObjectiveInterval>,
    ) -> Self {
        Self {
            id: id.into(),
            uncertainty_profile_id: uncertainty_profile_id.into(),
            bounds,
        }
    }
}

/// Invalid profiles, candidate identities or uncertainty intervals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustParetoError {
    EmptyObjectives,
    EmptyObjectiveName,
    DuplicateObjectiveName,
    EmptyProfileId,
    EmptyCandidateId,
    DuplicateCandidateId,
    UncertaintyProfileMismatch,
    ObjectiveCountMismatch,
    NonFiniteBound,
    ReversedInterval,
}

impl fmt::Display for RobustParetoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyObjectives => "at least one objective is required",
            Self::EmptyObjectiveName => "objective names must not be empty",
            Self::DuplicateObjectiveName => "objective names must be unique",
            Self::EmptyProfileId => "uncertainty profile identifier must not be empty",
            Self::EmptyCandidateId => "candidate identifiers must not be empty",
            Self::DuplicateCandidateId => "candidate identifiers must be unique",
            Self::UncertaintyProfileMismatch => {
                "candidate uncertainty profile does not match the requested profile"
            }
            Self::ObjectiveCountMismatch => {
                "each candidate must provide one interval per objective"
            }
            Self::NonFiniteBound => "uncertainty bounds must all be finite",
            Self::ReversedInterval => "an uncertainty interval has lower greater than upper",
        };
        f.write_str(message)
    }
}

impl std::error::Error for RobustParetoError {}

/// Return candidate indices not robustly dominated by any other candidate.
///
/// For a minimized objective, A is demonstrably no worse than B only when
/// A.upper <= B.lower. For a maximized objective, A.lower >= B.upper.
/// Robust dominance additionally requires strict separation on at least one
/// objective. Intersecting bounds therefore remain incomparable on that
/// dimension; no midpoint, arbitrary weight or implicit tie-break is used.
///
/// When all intervals are point-valued, this relation exactly reduces to
/// ordinary Pareto dominance. uncertainty_profile_id must identify the same
/// predeclared interval semantics for every candidate. Empty candidate sets
/// return an empty front after the objective/profile contract is validated.
///
/// This is a conservative interval-order relation, not a joint probability
/// model. It does not account for covariance unless that is already reflected
/// validly in the caller's bounds; do not label it a confidence guarantee without
/// a separately justified uncertainty procedure.
pub fn robust_pareto_front_indices(
    candidates: &[BoundedCandidate],
    objectives: &[ObjectiveSpec],
    uncertainty_profile_id: &str,
) -> Result<Vec<usize>, RobustParetoError> {
    validate_objectives(objectives)?;
    if uncertainty_profile_id.trim().is_empty() {
        return Err(RobustParetoError::EmptyProfileId);
    }
    validate_candidates(candidates, objectives.len(), uncertainty_profile_id)?;

    let mut front = Vec::new();
    for candidate_index in 0..candidates.len() {
        let is_dominated = candidates.iter().enumerate().any(|(other_index, other)| {
            other_index != candidate_index
                && robustly_dominates(other, &candidates[candidate_index], objectives)
        });
        if !is_dominated {
            front.push(candidate_index);
        }
    }
    Ok(front)
}

fn validate_objectives(objectives: &[ObjectiveSpec]) -> Result<(), RobustParetoError> {
    if objectives.is_empty() {
        return Err(RobustParetoError::EmptyObjectives);
    }

    let mut names = HashSet::with_capacity(objectives.len());
    for objective in objectives {
        let name = objective.name.trim();
        if name.is_empty() {
            return Err(RobustParetoError::EmptyObjectiveName);
        }
        if !names.insert(name) {
            return Err(RobustParetoError::DuplicateObjectiveName);
        }
    }
    Ok(())
}

fn validate_candidates(
    candidates: &[BoundedCandidate],
    objective_count: usize,
    expected_profile_id: &str,
) -> Result<(), RobustParetoError> {
    let mut ids = HashSet::with_capacity(candidates.len());
    for candidate in candidates {
        if candidate.id.trim().is_empty() {
            return Err(RobustParetoError::EmptyCandidateId);
        }
        if !ids.insert(candidate.id.as_str()) {
            return Err(RobustParetoError::DuplicateCandidateId);
        }
        if candidate.uncertainty_profile_id.trim().is_empty() {
            return Err(RobustParetoError::EmptyProfileId);
        }
        if candidate.uncertainty_profile_id != expected_profile_id {
            return Err(RobustParetoError::UncertaintyProfileMismatch);
        }
        if candidate.bounds.len() != objective_count {
            return Err(RobustParetoError::ObjectiveCountMismatch);
        }
        for interval in &candidate.bounds {
            if !interval.lower.is_finite() || !interval.upper.is_finite() {
                return Err(RobustParetoError::NonFiniteBound);
            }
            if interval.lower > interval.upper {
                return Err(RobustParetoError::ReversedInterval);
            }
        }
    }
    Ok(())
}

fn robustly_dominates(
    candidate: &BoundedCandidate,
    other: &BoundedCandidate,
    objectives: &[ObjectiveSpec],
) -> bool {
    let mut strictly_better = false;
    for ((left, right), objective) in candidate
        .bounds
        .iter()
        .zip(&other.bounds)
        .zip(objectives)
    {
        let (no_worse, better) = match objective.direction {
            ObjectiveDirection::Minimize => {
                (left.upper <= right.lower, left.upper < right.lower)
            }
            ObjectiveDirection::Maximize => {
                (left.lower >= right.upper, left.lower > right.upper)
            }
        };
        if !no_worse {
            return false;
        }
        strictly_better |= better;
    }
    strictly_better
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pareto::{pareto_front_indices, ParetoCandidate};

    const PROFILE: &str = "fixture-profile-v1";

    fn objectives() -> Vec<ObjectiveSpec> {
        vec![
            ObjectiveSpec::minimize("bill_of_materials_usd"),
            ObjectiveSpec::maximize("maximum_linear_spl_db"),
        ]
    }

    #[test]
    fn robustly_dominated_designs_are_removed_and_tradeoffs_remain() {
        let candidates = vec![
            BoundedCandidate::new(
                "balanced",
                PROFILE,
                vec![ObjectiveInterval::new(98.0, 102.0), ObjectiveInterval::new(109.0, 111.0)],
            ),
            BoundedCandidate::new(
                "low_cost",
                PROFILE,
                vec![ObjectiveInterval::new(80.0, 85.0), ObjectiveInterval::new(97.0, 101.0)],
            ),
            BoundedCandidate::new(
                "dominated",
                PROFILE,
                vec![ObjectiveInterval::new(105.0, 110.0), ObjectiveInterval::new(102.0, 106.0)],
            ),
            BoundedCandidate::new(
                "overlapping_uncertainty",
                PROFILE,
                vec![ObjectiveInterval::new(101.0, 108.0), ObjectiveInterval::new(108.0, 112.0)],
            ),
        ];

        assert_eq!(
            robust_pareto_front_indices(&candidates, &objectives(), PROFILE).unwrap(),
            vec![0, 1, 3]
        );
    }

    #[test]
    fn point_intervals_reproduce_classic_pareto_front() {
        let raw = vec![
            ParetoCandidate::new("balanced", vec![100.0, 110.0]),
            ParetoCandidate::new("worse_on_both", vec![120.0, 105.0]),
            ParetoCandidate::new("low_cost", vec![80.0, 100.0]),
            ParetoCandidate::new("high_output", vec![140.0, 115.0]),
            ParetoCandidate::new("dominated", vec![130.0, 103.0]),
        ];
        let bounded: Vec<_> = raw
            .iter()
            .map(|candidate| {
                BoundedCandidate::new(
                    candidate.id.clone(),
                    PROFILE,
                    candidate.values.iter().copied().map(ObjectiveInterval::point).collect(),
                )
            })
            .collect();

        let point_front = pareto_front_indices(&raw, &objectives()).unwrap();
        let robust_front =
            robust_pareto_front_indices(&bounded, &objectives(), PROFILE).unwrap();
        assert_eq!(robust_front, point_front);
    }

    #[test]
    fn overlapping_intervals_are_incomparable_not_equal() {
        let candidates = vec![
            BoundedCandidate::new(
                "a",
                PROFILE,
                vec![ObjectiveInterval::new(99.0, 101.0)],
            ),
            BoundedCandidate::new(
                "b",
                PROFILE,
                vec![ObjectiveInterval::new(100.0, 102.0)],
            ),
        ];
        let single_objective = [ObjectiveSpec::minimize("response_error")];

        assert_eq!(
            robust_pareto_front_indices(&candidates, &single_objective, PROFILE).unwrap(),
            vec![0, 1]
        );
    }

    #[test]
    fn touching_bounds_alone_are_not_strict_dominance() {
        let candidates = vec![
            BoundedCandidate::new("a", PROFILE, vec![ObjectiveInterval::new(0.0, 1.0)]),
            BoundedCandidate::new("b", PROFILE, vec![ObjectiveInterval::new(1.0, 2.0)]),
        ];
        let single_objective = [ObjectiveSpec::minimize("response_error")];

        assert_eq!(
            robust_pareto_front_indices(&candidates, &single_objective, PROFILE).unwrap(),
            vec![0, 1]
        );
    }

    #[test]
    fn empty_candidates_produce_empty_front_for_valid_contract() {
        assert_eq!(
            robust_pareto_front_indices(&[], &objectives(), PROFILE).unwrap(),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        let valid = BoundedCandidate::new(
            "valid",
            PROFILE,
            vec![ObjectiveInterval::point(100.0), ObjectiveInterval::point(110.0)],
        );

        assert_eq!(
            robust_pareto_front_indices(&[valid.clone()], &[], PROFILE),
            Err(RobustParetoError::EmptyObjectives)
        );
        assert_eq!(
            robust_pareto_front_indices(&[valid.clone()], &objectives(), " "),
            Err(RobustParetoError::EmptyProfileId)
        );
        assert_eq!(
            robust_pareto_front_indices(&[valid.clone()], &objectives(), "other-profile"),
            Err(RobustParetoError::UncertaintyProfileMismatch)
        );
        assert_eq!(
            robust_pareto_front_indices(
                &[BoundedCandidate::new("  ", PROFILE, valid.bounds.clone())],
                &objectives(),
                PROFILE,
            ),
            Err(RobustParetoError::EmptyCandidateId)
        );
        assert_eq!(
            robust_pareto_front_indices(
                &[valid.clone(), valid.clone()],
                &objectives(),
                PROFILE,
            ),
            Err(RobustParetoError::DuplicateCandidateId)
        );
        assert_eq!(
            robust_pareto_front_indices(
                &[BoundedCandidate::new(
                    "short",
                    PROFILE,
                    vec![ObjectiveInterval::point(1.0)],
                )],
                &objectives(),
                PROFILE,
            ),
            Err(RobustParetoError::ObjectiveCountMismatch)
        );
        assert_eq!(
            robust_pareto_front_indices(
                &[BoundedCandidate::new(
                    "nan",
                    PROFILE,
                    vec![ObjectiveInterval::new(f64::NAN, 1.0), ObjectiveInterval::point(1.0)],
                )],
                &objectives(),
                PROFILE,
            ),
            Err(RobustParetoError::NonFiniteBound)
        );
        assert_eq!(
            robust_pareto_front_indices(
                &[BoundedCandidate::new(
                    "reversed",
                    PROFILE,
                    vec![ObjectiveInterval::new(2.0, 1.0), ObjectiveInterval::point(1.0)],
                )],
                &objectives(),
                PROFILE,
            ),
            Err(RobustParetoError::ReversedInterval)
        );
        assert_eq!(
            robust_pareto_front_indices(
                &[BoundedCandidate::new(
                    "empty-profile",
                    " ",
                    valid.bounds.clone(),
                )],
                &objectives(),
                PROFILE,
            ),
            Err(RobustParetoError::EmptyProfileId)
        );
    }

    #[test]
    fn duplicate_objective_names_fail_closed() {
        let duplicate = [
            ObjectiveSpec::minimize("same"),
            ObjectiveSpec::maximize("same"),
        ];
        let candidate = BoundedCandidate::new(
            "candidate",
            PROFILE,
            vec![ObjectiveInterval::point(1.0), ObjectiveInterval::point(2.0)],
        );

        assert_eq!(
            robust_pareto_front_indices(&[candidate], &duplicate, PROFILE),
            Err(RobustParetoError::DuplicateObjectiveName)
        );
    }
}
