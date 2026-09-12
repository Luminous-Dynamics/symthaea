// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Domain-neutral deterministic Pareto ranking kernel.
//!
//! This crate intentionally knows nothing about candidate generation, fitness
//! weights, feasibility policy, or domain-specific objective meaning. Callers
//! supply validated objective vectors and explicit directions.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectiveSpec {
    pub name: String,
    pub direction: Direction,
}

impl ObjectiveSpec {
    pub fn new(name: impl Into<String>, direction: Direction) -> Result<Self, ParetoError> {
        let value = Self {
            name: name.into(),
            direction,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), ParetoError> {
        if self.name.trim().is_empty() {
            return Err(ParetoError::InvalidObjective(
                "objective name cannot be empty".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub values: Vec<f64>,
}

impl Point {
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ranking {
    /// Pareto rank for each input point. Rank zero is the non-dominated front.
    pub ranks: Vec<usize>,
    /// Input indices grouped by rank, preserving original input order inside
    /// every front.
    pub fronts: Vec<Vec<usize>>,
    /// NSGA-II-style crowding distance for each input point, computed only
    /// against members of the same front.
    pub crowding_distance: Vec<f64>,
}

impl Ranking {
    pub fn frontier(&self) -> &[usize] {
        self.fronts.first().map(Vec::as_slice).unwrap_or(&[])
    }
}

/// Rank finite objective vectors using deterministic non-dominated sorting.
pub fn rank(specs: &[ObjectiveSpec], points: &[Point]) -> Result<Ranking, ParetoError> {
    validate_inputs(specs, points)?;
    if points.is_empty() {
        return Ok(Ranking {
            ranks: Vec::new(),
            fronts: Vec::new(),
            crowding_distance: Vec::new(),
        });
    }

    let n = points.len();
    let mut domination_count = vec![0usize; n];
    let mut dominates_set = vec![Vec::<usize>::new(); n];

    for left in 0..n {
        for right in (left + 1)..n {
            match dominance(specs, &points[left], &points[right]) {
                Dominance::Left => {
                    dominates_set[left].push(right);
                    domination_count[right] += 1;
                }
                Dominance::Right => {
                    dominates_set[right].push(left);
                    domination_count[left] += 1;
                }
                Dominance::Neither => {}
            }
        }
    }

    let mut ranks = vec![usize::MAX; n];
    let mut fronts = Vec::new();
    let mut current: Vec<usize> = (0..n)
        .filter(|&index| domination_count[index] == 0)
        .collect();
    let mut current_rank = 0usize;

    while !current.is_empty() {
        current.sort_unstable();
        for &index in &current {
            ranks[index] = current_rank;
        }

        let mut next = Vec::new();
        for &index in &current {
            for &dominated in &dominates_set[index] {
                domination_count[dominated] = domination_count[dominated].saturating_sub(1);
                if domination_count[dominated] == 0 {
                    next.push(dominated);
                }
            }
        }
        next.sort_unstable();
        next.dedup();
        fronts.push(current);
        current = next;
        current_rank += 1;
    }

    if ranks.iter().any(|&value| value == usize::MAX) {
        return Err(ParetoError::Internal(
            "non-dominated sort left at least one point unranked".into(),
        ));
    }

    let mut crowding_distance = vec![0.0; n];
    for front in &fronts {
        assign_crowding(specs, points, front, &mut crowding_distance);
    }

    Ok(Ranking {
        ranks,
        fronts,
        crowding_distance,
    })
}

fn validate_inputs(specs: &[ObjectiveSpec], points: &[Point]) -> Result<(), ParetoError> {
    if specs.is_empty() {
        return Err(ParetoError::InvalidObjective(
            "at least one objective is required".into(),
        ));
    }

    let mut names = BTreeSet::new();
    for spec in specs {
        spec.validate()?;
        if !names.insert(spec.name.as_str()) {
            return Err(ParetoError::DuplicateObjective(spec.name.clone()));
        }
    }

    for (point_index, point) in points.iter().enumerate() {
        if point.values.len() != specs.len() {
            return Err(ParetoError::DimensionMismatch {
                point_index,
                expected: specs.len(),
                actual: point.values.len(),
            });
        }
        for (objective_index, value) in point.values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(ParetoError::NonFiniteValue {
                    point_index,
                    objective_index,
                    value,
                });
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dominance {
    Left,
    Right,
    Neither,
}

fn dominance(specs: &[ObjectiveSpec], left: &Point, right: &Point) -> Dominance {
    let mut left_strict = false;
    let mut right_strict = false;

    for ((spec, &left_value), &right_value) in specs
        .iter()
        .zip(left.values.iter())
        .zip(right.values.iter())
    {
        let ordering = match spec.direction {
            Direction::Minimize => left_value.total_cmp(&right_value),
            Direction::Maximize => right_value.total_cmp(&left_value),
        };
        match ordering {
            std::cmp::Ordering::Less => left_strict = true,
            std::cmp::Ordering::Greater => right_strict = true,
            std::cmp::Ordering::Equal => {}
        }
        if left_strict && right_strict {
            return Dominance::Neither;
        }
    }

    match (left_strict, right_strict) {
        (true, false) => Dominance::Left,
        (false, true) => Dominance::Right,
        _ => Dominance::Neither,
    }
}

fn assign_crowding(
    _specs: &[ObjectiveSpec],
    points: &[Point],
    front: &[usize],
    distances: &mut [f64],
) {
    if front.is_empty() {
        return;
    }
    if front.len() <= 2 {
        for &index in front {
            distances[index] = f64::INFINITY;
        }
        return;
    }

    for &index in front {
        distances[index] = 0.0;
    }

    let objective_count = points[front[0]].values.len();
    for objective_index in 0..objective_count {
        let mut ordered = front.to_vec();
        ordered.sort_by(|&left, &right| {
            points[left].values[objective_index]
                .total_cmp(&points[right].values[objective_index])
                .then(left.cmp(&right))
        });

        let first = ordered[0];
        let last = ordered[ordered.len() - 1];
        let min = points[first].values[objective_index];
        let max = points[last].values[objective_index];
        let range = max - min;

        // A constant objective contains no diversity information. In
        // particular, it must not award infinity to arbitrary endpoints that
        // happen to sort first/last only because of input order.
        if range.abs() <= f64::EPSILON {
            continue;
        }

        distances[first] = f64::INFINITY;
        distances[last] = f64::INFINITY;

        for window_index in 1..(ordered.len() - 1) {
            let current = ordered[window_index];
            if distances[current].is_infinite() {
                continue;
            }
            let previous = points[ordered[window_index - 1]].values[objective_index];
            let next = points[ordered[window_index + 1]].values[objective_index];
            distances[current] += (next - previous).abs() / range.abs();
        }
    }
}

#[derive(Debug, Error)]
pub enum ParetoError {
    #[error("invalid objective: {0}")]
    InvalidObjective(String),
    #[error("duplicate objective name {0:?}")]
    DuplicateObjective(String),
    #[error(
        "point {point_index} has {actual} objectives; expected exactly {expected}"
    )]
    DimensionMismatch {
        point_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "point {point_index} objective {objective_index} is non-finite: {value}"
    )]
    NonFiniteValue {
        point_index: usize,
        objective_index: usize,
        value: f64,
    },
    #[error("Pareto kernel internal error: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specs() -> Vec<ObjectiveSpec> {
        vec![
            ObjectiveSpec::new("cost", Direction::Minimize).unwrap(),
            ObjectiveSpec::new("lifetime", Direction::Maximize).unwrap(),
        ]
    }

    #[test]
    fn mixed_directions_produce_expected_fronts() {
        // A and B trade cost for lifetime. C is worse than A on both. D is
        // cheaper than both but short-lived, so it also belongs on frontier 0.
        let points = vec![
            Point::new(vec![10.0, 8.0]), // A
            Point::new(vec![12.0, 10.0]), // B
            Point::new(vec![13.0, 7.0]), // C
            Point::new(vec![8.0, 5.0]), // D
        ];
        let ranking = rank(&specs(), &points).unwrap();
        assert_eq!(ranking.fronts[0], vec![0, 1, 3]);
        assert_eq!(ranking.fronts[1], vec![2]);
        assert_eq!(ranking.ranks, vec![0, 0, 1, 0]);
    }

    #[test]
    fn equal_points_do_not_dominate_each_other() {
        let points = vec![Point::new(vec![1.0, 2.0]), Point::new(vec![1.0, 2.0])];
        let ranking = rank(&specs(), &points).unwrap();
        assert_eq!(ranking.frontier(), &[0, 1]);
        assert_eq!(ranking.ranks, vec![0, 0]);
    }

    #[test]
    fn all_minimize_matches_simple_design_space_dominance() {
        let specs = vec![
            ObjectiveSpec::new("mass", Direction::Minimize).unwrap(),
            ObjectiveSpec::new("cost", Direction::Minimize).unwrap(),
            ObjectiveSpec::new("dose", Direction::Minimize).unwrap(),
        ];
        let points = vec![
            Point::new(vec![10.0, 100.0, 0.1]),
            Point::new(vec![12.0, 110.0, 0.2]),
            Point::new(vec![8.0, 130.0, 0.08]),
        ];
        let ranking = rank(&specs, &points).unwrap();
        assert_eq!(ranking.fronts[0], vec![0, 2]);
        assert_eq!(ranking.fronts[1], vec![1]);
    }

    #[test]
    fn crowding_marks_real_boundaries_and_normalizes_interior_distance() {
        // All three points are non-dominated: lower cost trades against higher
        // lifetime. The middle point is interior on both objectives.
        let points = vec![
            Point::new(vec![0.0, 0.0]),
            Point::new(vec![1.0, 1.0]),
            Point::new(vec![3.0, 3.0]),
        ];
        let ranking = rank(&specs(), &points).unwrap();
        assert_eq!(ranking.frontier(), &[0, 1, 2]);
        assert!(ranking.crowding_distance[0].is_infinite());
        assert!(ranking.crowding_distance[2].is_infinite());
        assert!((ranking.crowding_distance[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn constant_objective_does_not_create_arbitrary_crowding_boundaries() {
        let specs = vec![
            ObjectiveSpec::new("constant", Direction::Minimize).unwrap(),
            ObjectiveSpec::new("tradeoff_a", Direction::Minimize).unwrap(),
            ObjectiveSpec::new("tradeoff_b", Direction::Maximize).unwrap(),
        ];
        let points = vec![
            Point::new(vec![1.0, 0.0, 0.0]),
            Point::new(vec![1.0, 1.0, 1.0]),
            Point::new(vec![1.0, 2.0, 2.0]),
        ];
        let ranking = rank(&specs, &points).unwrap();
        assert_eq!(ranking.frontier(), &[0, 1, 2]);
        assert!(ranking.crowding_distance[0].is_infinite());
        assert!(ranking.crowding_distance[2].is_infinite());
        // Only the two informative tradeoff objectives contribute: 1 + 1.
        assert!((ranking.crowding_distance[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        assert!(rank(&[], &[Point::new(vec![])]).is_err());
        assert!(rank(&specs(), &[Point::new(vec![1.0])]).is_err());
        assert!(rank(&specs(), &[Point::new(vec![f64::NAN, 1.0])]).is_err());

        let duplicates = vec![
            ObjectiveSpec::new("x", Direction::Minimize).unwrap(),
            ObjectiveSpec::new("x", Direction::Maximize).unwrap(),
        ];
        assert!(rank(&duplicates, &[Point::new(vec![1.0, 2.0])]).is_err());
    }

    #[test]
    fn empty_population_is_valid_and_deterministic() {
        let ranking = rank(&specs(), &[]).unwrap();
        assert!(ranking.ranks.is_empty());
        assert!(ranking.fronts.is_empty());
        assert!(ranking.crowding_distance.is_empty());
    }
}
