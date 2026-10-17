// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic Pareto-front extraction for acoustic design exploration.
//!
//! Keep objectives as separate dimensions rather than combining them into an
//! arbitrary weighted "speaker score". Objective values are ordered exactly as
//! their ObjectiveSpec entries. This module ranks candidates; it does not
//! decide whether measurements, simulations, hard constraints, or evidence are
//! trustworthy. Apply eligibility and provenance gates before ranking.

use std::collections::HashSet;
use std::fmt;

/// Whether a lower or higher value is preferred for one objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveDirection {
    Minimize,
    Maximize,
}

/// A named objective and its preferred direction.
///
/// Names are descriptive metadata. ParetoCandidate::values must follow the
/// same order as the objective specification slice passed to
/// pareto_front_indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectiveSpec {
    pub name: String,
    pub direction: ObjectiveDirection,
}

impl ObjectiveSpec {
    /// Create an objective for which lower values are preferred.
    pub fn minimize(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            direction: ObjectiveDirection::Minimize,
        }
    }

    /// Create an objective for which higher values are preferred.
    pub fn maximize(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            direction: ObjectiveDirection::Maximize,
        }
    }
}

/// One fully scored, eligible design candidate.
///
/// Values must be finite and use the units/definitions frozen by the calling
/// benchmark profile. Hard-constraint failures and unscored solver outcomes
/// belong in the caller's evaluation record; do not invent values to force
/// them into a Pareto comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct ParetoCandidate {
    pub id: String,
    pub values: Vec<f64>,
}

impl ParetoCandidate {
    /// Construct a candidate with a stable, caller-supplied identifier.
    pub fn new(id: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            id: id.into(),
            values,
        }
    }
}

/// Invalid objective specifications or candidate data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParetoError {
    EmptyObjectives,
    EmptyObjectiveName,
    DuplicateObjectiveName,
    EmptyCandidateId,
    DuplicateCandidateId,
    ObjectiveCountMismatch,
    NonFiniteObjectiveValue,
}

impl fmt::Display for ParetoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyObjectives => "at least one objective is required",
            Self::EmptyObjectiveName => "objective names must not be empty",
            Self::DuplicateObjectiveName => "objective names must be unique",
            Self::EmptyCandidateId => "candidate identifiers must not be empty",
            Self::DuplicateCandidateId => "candidate identifiers must be unique",
            Self::ObjectiveCountMismatch => {
                "each candidate must provide one value per objective"
            }
            Self::NonFiniteObjectiveValue => "objective values must all be finite",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ParetoError {}

/// Return indices of the non-dominated candidates, in original input order.
///
/// Candidate a dominates candidate b when a is no worse on every objective
/// and strictly better on at least one. Exact ties are retained: neither tied
/// candidate dominates the other. Objective direction is explicit, and all
/// data is validated before any result is returned.
///
/// This is an unweighted Pareto filter, not a universal quality score. It does
/// not include uncertainty, confidence bounds, feasibility constraints, or
/// preferences among trade-offs; callers must define and retain those separately.
/// Complexity is O(candidate_count² × objective_count), suitable for design
/// batches where auditability and deterministic behavior matter more than scale.
pub fn pareto_front_indices(
    candidates: &[ParetoCandidate],
    objectives: &[ObjectiveSpec],
) -> Result<Vec<usize>, ParetoError> {
    validate_objectives(objectives)?;
    validate_candidates(candidates, objectives.len())?;

    let mut front = Vec::new();
    for candidate_index in 0..candidates.len() {
        let is_dominated = candidates.iter().enumerate().any(|(other_index, other)| {
            other_index != candidate_index
                && dominates(other, &candidates[candidate_index], objectives)
        });
        if !is_dominated {
            front.push(candidate_index);
        }
    }
    Ok(front)
}

fn validate_objectives(objectives: &[ObjectiveSpec]) -> Result<(), ParetoError> {
    if objectives.is_empty() {
        return Err(ParetoError::EmptyObjectives);
    }

    let mut names = HashSet::with_capacity(objectives.len());
    for objective in objectives {
        let name = objective.name.trim();
        if name.is_empty() {
            return Err(ParetoError::EmptyObjectiveName);
        }
        if !names.insert(name) {
            return Err(ParetoError::DuplicateObjectiveName);
        }
    }
    Ok(())
}

fn validate_candidates(
    candidates: &[ParetoCandidate],
    objective_count: usize,
) -> Result<(), ParetoError> {
    let mut ids = HashSet::with_capacity(candidates.len());
    for candidate in candidates {
        if candidate.id.trim().is_empty() {
            return Err(ParetoError::EmptyCandidateId);
        }
        if !ids.insert(candidate.id.as_str()) {
            return Err(ParetoError::DuplicateCandidateId);
        }
        if candidate.values.len() != objective_count {
            return Err(ParetoError::ObjectiveCountMismatch);
        }
        if candidate.values.iter().any(|value| !value.is_finite()) {
            return Err(ParetoError::NonFiniteObjectiveValue);
        }
    }
    Ok(())
}

fn dominates(
    candidate: &ParetoCandidate,
    other: &ParetoCandidate,
    objectives: &[ObjectiveSpec],
) -> bool {
    let mut strictly_better = false;
    for ((left, right), objective) in candidate
        .values
        .iter()
        .zip(&other.values)
        .zip(objectives)
    {
        let (no_worse, better) = match objective.direction {
            ObjectiveDirection::Minimize => (left <= right, left < right),
            ObjectiveDirection::Maximize => (left >= right, left > right),
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

    fn objectives() -> Vec<ObjectiveSpec> {
        vec![
            ObjectiveSpec::minimize("bill_of_materials_usd"),
            ObjectiveSpec::maximize("maximum_linear_spl_db"),
        ]
    }

    #[test]
    fn removes_dominated_candidates_but_preserves_tradeoffs() {
        let candidates = vec![
            ParetoCandidate::new("balanced", vec![100.0, 110.0]),
            ParetoCandidate::new("worse_on_both", vec![120.0, 105.0]),
            ParetoCandidate::new("low_cost", vec![80.0, 100.0]),
            ParetoCandidate::new("high_output", vec![140.0, 115.0]),
            ParetoCandidate::new("dominated_by_balanced", vec![130.0, 103.0]),
        ];

        assert_eq!(
            pareto_front_indices(&candidates, &objectives()).unwrap(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn exact_ties_are_retained_in_input_order() {
        let candidates = vec![
            ParetoCandidate::new("unit-a", vec![100.0, 110.0]),
            ParetoCandidate::new("unit-b", vec![100.0, 110.0]),
            ParetoCandidate::new("dominated", vec![120.0, 105.0]),
        ];

        assert_eq!(
            pareto_front_indices(&candidates, &objectives()).unwrap(),
            vec![0, 1]
        );
    }

    #[test]
    fn empty_candidate_set_has_an_empty_front() {
        assert_eq!(
            pareto_front_indices(&[], &objectives()).unwrap(),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        let valid_candidate = ParetoCandidate::new("valid", vec![100.0, 110.0]);

        assert_eq!(
            pareto_front_indices(&[valid_candidate.clone()], &[]),
            Err(ParetoError::EmptyObjectives)
        );
        assert_eq!(
            pareto_front_indices(
                &[valid_candidate.clone()],
                &[ObjectiveSpec::minimize("  "), ObjectiveSpec::maximize("spl")]
            ),
            Err(ParetoError::EmptyObjectiveName)
        );
        assert_eq!(
            pareto_front_indices(
                &[valid_candidate.clone()],
                &[ObjectiveSpec::minimize("cost"), ObjectiveSpec::maximize("cost")]
            ),
            Err(ParetoError::DuplicateObjectiveName)
        );
        assert_eq!(
            pareto_front_indices(
                &[ParetoCandidate::new("  ", vec![100.0, 110.0])],
                &objectives()
            ),
            Err(ParetoError::EmptyCandidateId)
        );
        assert_eq!(
            pareto_front_indices(
                &[
                    valid_candidate.clone(),
                    ParetoCandidate::new("valid", vec![80.0, 100.0]),
                ],
                &objectives()
            ),
            Err(ParetoError::DuplicateCandidateId)
        );
        assert_eq!(
            pareto_front_indices(
                &[ParetoCandidate::new("wrong-shape", vec![100.0])],
                &objectives()
            ),
            Err(ParetoError::ObjectiveCountMismatch)
        );
        assert_eq!(
            pareto_front_indices(
                &[ParetoCandidate::new("non-finite", vec![100.0, f64::NAN])],
                &objectives()
            ),
            Err(ParetoError::NonFiniteObjectiveValue)
        );
    }
}
