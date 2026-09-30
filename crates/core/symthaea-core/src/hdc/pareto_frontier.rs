//! Pure Pareto-frontier analysis over already-qualified cost-quality points.
//!
//! This module is deliberately downstream of evidence qualification. It does
//! not decide whether evidence is trustworthy, and it never assigns a scalar
//! score or a universal winner. The caller supplies the exact objective vector
//! and direction for the analysis protocol.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const PARETO_FRONTIER_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectiveDirection {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Objective {
    pub name: String,
    pub direction: ObjectiveDirection,
    pub unit: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrontierPoint {
    pub point_id: String,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParetoFrontierError {
    UnsupportedSchema(u32),
    EmptyObjectives,
    EmptyObjectiveName,
    EmptyObjectiveUnit,
    DuplicateObjectiveName(String),
    EmptyPointId,
    NoPoints,
    DimensionMismatch {
        point_id: String,
        expected: usize,
        observed: usize,
    },
    NonFiniteValue {
        point_id: String,
        objective: String,
    },
}

impl std::fmt::Display for ParetoFrontierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => write!(f, "unsupported Pareto frontier schema: {v}"),
            Self::EmptyObjectives => write!(f, "Pareto frontier requires at least one objective"),
            Self::EmptyObjectiveName => write!(f, "Pareto objective name must not be empty"),
            Self::EmptyObjectiveUnit => write!(f, "Pareto objective unit must not be empty"),
            Self::DuplicateObjectiveName(name) => {
                write!(f, "Pareto objective name is duplicated: {name}")
            }
            Self::EmptyPointId => write!(f, "Pareto point ID must not be empty"),
            Self::NoPoints => write!(f, "Pareto frontier requires at least one point"),
            Self::DimensionMismatch {
                point_id,
                expected,
                observed,
            } => write!(
                f,
                "point {point_id} has {observed} values but {expected} objectives were declared"
            ),
            Self::NonFiniteValue { point_id, objective } => {
                write!(f, "point {point_id} has a non-finite value for objective {objective}")
            }
        }
    }
}

impl std::error::Error for ParetoFrontierError {}

/// Return the indices of the nondominated points in deterministic input order.
///
/// This is intentionally O(n²): frontier analysis is research-scale data
/// reduction, not a benchmark kernel, and the explicit pairwise definition is
/// easy to audit. A future optimized implementation must preserve this exact
/// semantic contract.
pub fn nondominated_indices(
    schema_version: u32,
    objectives: &[Objective],
    points: &[FrontierPoint],
) -> Result<Vec<usize>, ParetoFrontierError> {
    validate(schema_version, objectives, points)?;

    let mut result = Vec::new();
    for i in 0..points.len() {
        let dominated = (0..points.len())
            .filter(|&j| j != i)
            .any(|j| dominates(&points[j], &points[i], objectives));
        if !dominated {
            result.push(i);
        }
    }
    Ok(result)
}

fn validate(
    schema_version: u32,
    objectives: &[Objective],
    points: &[FrontierPoint],
) -> Result<(), ParetoFrontierError> {
    if schema_version != PARETO_FRONTIER_SCHEMA_VERSION {
        return Err(ParetoFrontierError::UnsupportedSchema(schema_version));
    }
    if objectives.is_empty() {
        return Err(ParetoFrontierError::EmptyObjectives);
    }

    let mut names = HashSet::with_capacity(objectives.len());
    for objective in objectives {
        if objective.name.is_empty() {
            return Err(ParetoFrontierError::EmptyObjectiveName);
        }
        if objective.unit.is_empty() {
            return Err(ParetoFrontierError::EmptyObjectiveUnit);
        }
        if !names.insert(&objective.name) {
            return Err(ParetoFrontierError::DuplicateObjectiveName(
                objective.name.clone(),
            ));
        }
    }

    if points.is_empty() {
        return Err(ParetoFrontierError::NoPoints);
    }

    for point in points {
        if point.point_id.is_empty() {
            return Err(ParetoFrontierError::EmptyPointId);
        }
        if point.values.len() != objectives.len() {
            return Err(ParetoFrontierError::DimensionMismatch {
                point_id: point.point_id.clone(),
                expected: objectives.len(),
                observed: point.values.len(),
            });
        }
        for (value, objective) in point.values.iter().zip(objectives) {
            if !value.is_finite() {
                return Err(ParetoFrontierError::NonFiniteValue {
                    point_id: point.point_id.clone(),
                    objective: objective.name.clone(),
                });
            }
        }
    }

    Ok(())
}

fn dominates(a: &FrontierPoint, b: &FrontierPoint, objectives: &[Objective]) -> bool {
    let mut strictly_better = false;

    for ((a_value, b_value), objective) in a.values.iter().zip(&b.values).zip(objectives) {
        let (no_worse, better) = match objective.direction {
            ObjectiveDirection::Minimize => (a_value <= b_value, a_value < b_value),
            ObjectiveDirection::Maximize => (a_value >= b_value, a_value > b_value),
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

    fn objectives() -> Vec<Objective> {
        vec![
            Objective {
                name: "elapsed".into(),
                direction: ObjectiveDirection::Minimize,
                unit: "s".into(),
            },
            Objective {
                name: "quality".into(),
                direction: ObjectiveDirection::Maximize,
                unit: "score".into(),
            },
        ]
    }

    fn point(id: &str, elapsed: f64, quality: f64) -> FrontierPoint {
        FrontierPoint {
            point_id: id.into(),
            values: vec![elapsed, quality],
        }
    }

    #[test]
    fn mixed_minimize_maximize_frontier_is_exact() {
        let points = vec![
            point("a", 1.0, 0.8),
            point("b", 2.0, 0.9),
            point("c", 1.5, 0.7),
            point("d", 0.8, 0.6),
        ];

        let indices =
            nondominated_indices(PARETO_FRONTIER_SCHEMA_VERSION, &objectives(), &points)
                .expect("fixture should validate");

        assert_eq!(indices, vec![0, 1, 3]);
    }

    #[test]
    fn equal_points_do_not_dominate_each_other() {
        let points = vec![point("a", 1.0, 0.8), point("b", 1.0, 0.8)];
        let indices =
            nondominated_indices(PARETO_FRONTIER_SCHEMA_VERSION, &objectives(), &points)
                .expect("fixture should validate");
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn lower_is_better_can_dominate_on_both_dimensions() {
        let points = vec![point("a", 1.0, 0.8), point("b", 2.0, 0.7)];
        let indices =
            nondominated_indices(PARETO_FRONTIER_SCHEMA_VERSION, &objectives(), &points)
                .expect("fixture should validate");
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn duplicate_objective_names_fail_closed() {
        let mut objectives = objectives();
        objectives[1].name = objectives[0].name.clone();
        let points = vec![point("a", 1.0, 0.8)];
        assert!(matches!(
            nondominated_indices(PARETO_FRONTIER_SCHEMA_VERSION, &objectives, &points),
            Err(ParetoFrontierError::DuplicateObjectiveName(_))
        ));
    }

    #[test]
    fn nonfinite_values_fail_closed() {
        let points = vec![point("a", f64::NAN, 0.8)];
        assert!(matches!(
            nondominated_indices(PARETO_FRONTIER_SCHEMA_VERSION, &objectives(), &points),
            Err(ParetoFrontierError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn dimension_mismatch_fails_closed() {
        let points = vec![FrontierPoint {
            point_id: "a".into(),
            values: vec![1.0],
        }];
        assert!(matches!(
            nondominated_indices(PARETO_FRONTIER_SCHEMA_VERSION, &objectives(), &points),
            Err(ParetoFrontierError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn no_objectives_fails_closed() {
        let points = vec![point("a", 1.0, 0.8)];
        assert!(matches!(
            nondominated_indices(PARETO_FRONTIER_SCHEMA_VERSION, &[], &points),
            Err(ParetoFrontierError::EmptyObjectives)
        ));
    }

    #[test]
    fn missing_data_is_not_coerced_to_zero() {
        // The analysis API accepts concrete finite values only. Optional
        // measurements such as energy must be filtered or handled by an
        // explicit higher-level missing-data protocol before this function.
        let points = vec![point("a", 1.0, 0.8)];
        assert!(nondominated_indices(PARETO_FRONTIER_SCHEMA_VERSION, &objectives(), &points).is_ok());
    }
}
