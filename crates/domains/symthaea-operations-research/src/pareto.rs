// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic Pareto ranking for multi-objective engineering decisions.
//!
//! This module intentionally does not collapse multiple objectives into a weighted
//! scalar score. It identifies non-dominated candidates and successive Pareto fronts
//! while preserving objective direction explicitly.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Whether lower or higher values are preferred for one objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveDirection {
    Minimize,
    Maximize,
}

/// One named optimization objective.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Objective {
    pub name: String,
    pub direction: ObjectiveDirection,
}

impl Objective {
    pub fn new(name: impl Into<String>, direction: ObjectiveDirection) -> Self {
        Self {
            name: name.into(),
            direction,
        }
    }
}

/// A candidate evaluated against every objective in declaration order.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub id: String,
    pub values: Vec<f64>,
}

impl Candidate {
    pub fn new(id: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            id: id.into(),
            values,
        }
    }
}

/// Deterministic non-dominated sorting result.
///
/// `fronts[0]` is the Pareto frontier. Later fronts are obtained by repeatedly
/// removing all candidates in the preceding front.
#[derive(Debug, Clone, PartialEq)]
pub struct ParetoRanking {
    pub fronts: Vec<Vec<Candidate>>,
}

impl ParetoRanking {
    pub fn first_front(&self) -> &[Candidate] {
        self.fronts.first().map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn rank_of(&self, candidate_id: &str) -> Option<usize> {
        self.fronts.iter().position(|front| {
            front
                .iter()
                .any(|candidate| candidate.id == candidate_id)
        })
    }
}

/// Rank candidates into deterministic Pareto fronts.
///
/// Candidate identity is required to be unique. Inputs are sorted by ID before
/// ranking so the result is independent of caller iteration order.
pub fn rank_pareto(
    objectives: &[Objective],
    candidates: &[Candidate],
) -> Result<ParetoRanking, ParetoError> {
    if objectives.is_empty() {
        return Err(ParetoError::NoObjectives);
    }

    let mut ordered = candidates.to_vec();
    ordered.sort_by(|left, right| left.id.cmp(&right.id));

    let mut ids = BTreeSet::new();
    for candidate in &ordered {
        if !ids.insert(candidate.id.clone()) {
            return Err(ParetoError::DuplicateCandidate(candidate.id.clone()));
        }
        if candidate.values.len() != objectives.len() {
            return Err(ParetoError::DimensionMismatch {
                candidate: candidate.id.clone(),
                expected: objectives.len(),
                actual: candidate.values.len(),
            });
        }
        for (index, value) in candidate.values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(ParetoError::NonFiniteValue {
                    candidate: candidate.id.clone(),
                    objective: objectives[index].name.clone(),
                    value,
                });
            }
        }
    }

    if ordered.is_empty() {
        return Ok(ParetoRanking { fronts: Vec::new() });
    }

    let count = ordered.len();
    let mut dominates_set = vec![Vec::<usize>::new(); count];
    let mut dominated_by_count = vec![0usize; count];

    for i in 0..count {
        for j in (i + 1)..count {
            if dominates(&ordered[i], &ordered[j], objectives) {
                dominates_set[i].push(j);
                dominated_by_count[j] += 1;
            } else if dominates(&ordered[j], &ordered[i], objectives) {
                dominates_set[j].push(i);
                dominated_by_count[i] += 1;
            }
        }
    }

    let mut current: Vec<usize> = dominated_by_count
        .iter()
        .enumerate()
        .filter_map(|(index, count)| (*count == 0).then_some(index))
        .collect();
    let mut fronts = Vec::new();
    let mut ranked = 0usize;

    while !current.is_empty() {
        current.sort_by(|left, right| ordered[*left].id.cmp(&ordered[*right].id));
        fronts.push(
            current
                .iter()
                .map(|index| ordered[*index].clone())
                .collect(),
        );
        ranked += current.len();

        let mut next = Vec::new();
        for index in current {
            for dominated in &dominates_set[index] {
                debug_assert!(dominated_by_count[*dominated] > 0);
                dominated_by_count[*dominated] -= 1;
                if dominated_by_count[*dominated] == 0 {
                    next.push(*dominated);
                }
            }
        }
        current = next;
    }

    debug_assert_eq!(ranked, ordered.len());
    Ok(ParetoRanking { fronts })
}

/// Return true only when `left` is no worse on every objective and strictly
/// better on at least one objective.
pub fn dominates(left: &Candidate, right: &Candidate, objectives: &[Objective]) -> bool {
    if left.values.len() != objectives.len() || right.values.len() != objectives.len() {
        return false;
    }

    let mut strictly_better = false;
    for ((left_value, right_value), objective) in left
        .values
        .iter()
        .zip(&right.values)
        .zip(objectives)
    {
        let (no_worse, better) = match objective.direction {
            ObjectiveDirection::Minimize => (left_value <= right_value, left_value < right_value),
            ObjectiveDirection::Maximize => (left_value >= right_value, left_value > right_value),
        };
        if !no_worse {
            return false;
        }
        strictly_better |= better;
    }
    strictly_better
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParetoError {
    NoObjectives,
    DuplicateCandidate(String),
    DimensionMismatch {
        candidate: String,
        expected: usize,
        actual: usize,
    },
    NonFiniteValue {
        candidate: String,
        objective: String,
        value: f64,
    },
}

impl fmt::Display for ParetoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoObjectives => write!(formatter, "at least one Pareto objective is required"),
            Self::DuplicateCandidate(id) => write!(formatter, "duplicate candidate id {id}"),
            Self::DimensionMismatch {
                candidate,
                expected,
                actual,
            } => write!(
                formatter,
                "candidate {candidate} has {actual} objective values; expected {expected}"
            ),
            Self::NonFiniteValue {
                candidate,
                objective,
                value,
            } => write!(
                formatter,
                "candidate {candidate} has non-finite value {value} for objective {objective}"
            ),
        }
    }
}

impl Error for ParetoError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn objectives() -> Vec<Objective> {
        vec![
            Objective::new("cost", ObjectiveDirection::Minimize),
            Objective::new("resilience", ObjectiveDirection::Maximize),
        ]
    }

    #[test]
    fn identifies_tradeoff_front_without_weighted_score() {
        let candidates = vec![
            Candidate::new("cheap", vec![1.0, 4.0]),
            Candidate::new("resilient", vec![4.0, 9.0]),
            Candidate::new("dominated", vec![5.0, 3.0]),
        ];
        let ranking = rank_pareto(&objectives(), &candidates).unwrap();
        let ids: Vec<&str> = ranking
            .first_front()
            .iter()
            .map(|candidate| candidate.id.as_str())
            .collect();
        assert_eq!(ids, vec!["cheap", "resilient"]);
        assert_eq!(ranking.rank_of("dominated"), Some(1));
    }

    #[test]
    fn ranking_is_independent_of_input_order() {
        let a = vec![
            Candidate::new("b", vec![2.0, 8.0]),
            Candidate::new("a", vec![1.0, 5.0]),
            Candidate::new("c", vec![3.0, 4.0]),
        ];
        let mut b = a.clone();
        b.reverse();
        assert_eq!(
            rank_pareto(&objectives(), &a).unwrap(),
            rank_pareto(&objectives(), &b).unwrap()
        );
    }

    #[test]
    fn equal_candidates_do_not_dominate_each_other() {
        let objectives = objectives();
        let a = Candidate::new("a", vec![2.0, 7.0]);
        let b = Candidate::new("b", vec![2.0, 7.0]);
        assert!(!dominates(&a, &b, &objectives));
        assert!(!dominates(&b, &a, &objectives));
        assert_eq!(rank_pareto(&objectives, &[a, b]).unwrap().first_front().len(), 2);
    }

    #[test]
    fn successive_fronts_are_computed() {
        let candidates = vec![
            Candidate::new("a", vec![1.0, 9.0]),
            Candidate::new("b", vec![2.0, 8.0]),
            Candidate::new("c", vec![3.0, 7.0]),
        ];
        let ranking = rank_pareto(&objectives(), &candidates).unwrap();
        assert_eq!(ranking.fronts.len(), 3);
        assert_eq!(ranking.fronts[0][0].id, "a");
        assert_eq!(ranking.fronts[1][0].id, "b");
        assert_eq!(ranking.fronts[2][0].id, "c");
    }

    #[test]
    fn dimension_mismatch_fails_closed() {
        let error = rank_pareto(&objectives(), &[Candidate::new("bad", vec![1.0])]).unwrap_err();
        assert!(matches!(error, ParetoError::DimensionMismatch { .. }));
    }

    #[test]
    fn non_finite_input_is_rejected() {
        let error = rank_pareto(
            &objectives(),
            &[Candidate::new("bad", vec![f64::NAN, 1.0])],
        )
        .unwrap_err();
        assert!(matches!(error, ParetoError::NonFiniteValue { .. }));
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let candidates = vec![
            Candidate::new("same", vec![1.0, 2.0]),
            Candidate::new("same", vec![2.0, 3.0]),
        ];
        assert!(matches!(
            rank_pareto(&objectives(), &candidates),
            Err(ParetoError::DuplicateCandidate(id)) if id == "same"
        ));
    }

    #[test]
    fn empty_candidate_set_is_valid() {
        assert!(rank_pareto(&objectives(), &[]).unwrap().fronts.is_empty());
    }
}
