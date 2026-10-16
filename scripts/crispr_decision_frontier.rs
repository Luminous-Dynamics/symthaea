//! Dependency-free Pareto frontier for CRISPR-OT decision-policy comparisons.
//!
//! Compile and run without Cargo or network access:
//!   rustc --edition=2024 --test scripts/crispr_decision_frontier.rs -o /tmp/crispr-frontier-tests
//!   /tmp/crispr-frontier-tests
//!
//! The frontier refuses implicit scalarization. Every policy must have the same
//! complete, finite objective vector; not-estimable objectives must be handled
//! explicitly by the benchmark report rather than zero-filled.

use std::collections::HashSet;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectiveDirection {
    Maximize,
    Minimize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Objective {
    pub name: String,
    pub direction: ObjectiveDirection,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyVector {
    pub policy_id: String,
    /// Values use original metric units; direction is defined by Objective.
    pub values: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FrontierError {
    NoObjectives,
    DuplicateObjectiveName(String),
    DuplicatePolicyId(String),
    ObjectiveCountMismatch {
        policy_id: String,
        expected: usize,
        actual: usize,
    },
    NonFiniteValue {
        policy_id: String,
        objective: String,
        value: f64,
    },
}

impl fmt::Display for FrontierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoObjectives => write!(f, "at least one objective is required"),
            Self::DuplicateObjectiveName(name) => {
                write!(f, "duplicate objective name: {name}")
            }
            Self::DuplicatePolicyId(id) => write!(f, "duplicate policy id: {id}"),
            Self::ObjectiveCountMismatch {
                policy_id,
                expected,
                actual,
            } => write!(
                f,
                "policy {policy_id} has {actual} objectives; expected {expected}"
            ),
            Self::NonFiniteValue {
                policy_id,
                objective,
                value,
            } => write!(
                f,
                "policy {policy_id} has non-finite value {value} for objective {objective}"
            ),
        }
    }
}

impl std::error::Error for FrontierError {}

fn validates<'a>(
    objectives: &'a [Objective],
    policies: &[PolicyVector],
) -> Result<(), FrontierError> {
    if objectives.is_empty() {
        return Err(FrontierError::NoObjectives);
    }

    let mut objective_names = HashSet::with_capacity(objectives.len());
    for objective in objectives {
        if !objective_names.insert(objective.name.as_str()) {
            return Err(FrontierError::DuplicateObjectiveName(
                objective.name.clone(),
            ));
        }
    }

    let mut policy_ids = HashSet::with_capacity(policies.len());
    for policy in policies {
        if !policy_ids.insert(policy.policy_id.as_str()) {
            return Err(FrontierError::DuplicatePolicyId(
                policy.policy_id.clone(),
            ));
        }
        if policy.values.len() != objectives.len() {
            return Err(FrontierError::ObjectiveCountMismatch {
                policy_id: policy.policy_id.clone(),
                expected: objectives.len(),
                actual: policy.values.len(),
            });
        }
        for (objective, value) in objectives.iter().zip(policy.values.iter()) {
            if !value.is_finite() {
                return Err(FrontierError::NonFiniteValue {
                    policy_id: policy.policy_id.clone(),
                    objective: objective.name.clone(),
                    value: *value,
                });
            }
        }
    }

    Ok(())
}

/// True if policy A is no worse than policy B on every objective and strictly
/// better on at least one. Exact comparison is deliberate: the caller should
/// round/pin metric values before frontier calculation if that is part of the
/// preregistered report contract; this function does not invent a tolerance.
pub fn dominates(
    objectives: &[Objective],
    policy_a: &PolicyVector,
    policy_b: &PolicyVector,
) -> bool {
    if objectives.len() != policy_a.values.len()
        || objectives.len() != policy_b.values.len()
    {
        return false;
    }

    let mut strictly_better = false;
    for ((objective, a), b) in objectives
        .iter()
        .zip(policy_a.values.iter())
        .zip(policy_b.values.iter())
    {
        let (no_worse, strictly) = match objective.direction {
            ObjectiveDirection::Maximize => (a >= b, a > b),
            ObjectiveDirection::Minimize => (a <= b, a < b),
        };
        if !no_worse {
            return false;
        }
        strictly_better |= strictly;
    }
    strictly_better
}

/// Return sorted IDs for the nondominated policies under the declared objective
/// directions. No weighted sum or universal "safety score" is constructed.
/// Empty policy input returns an empty frontier; invalid vectors fail closed.
pub fn pareto_frontier(
    objectives: &[Objective],
    policies: &[PolicyVector],
) -> Result<Vec<String>, FrontierError> {
    validates(objectives, policies)?;

    let mut frontier = Vec::new();
    for (i, policy) in policies.iter().enumerate() {
        let is_dominated = policies
            .iter()
            .enumerate()
            .any(|(j, other)| i != j && dominates(objectives, other, policy));
        if !is_dominated {
            frontier.push(policy.policy_id.clone());
        }
    }
    frontier.sort();
    Ok(frontier)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn objective(name: &str, direction: ObjectiveDirection) -> Objective {
        Objective {
            name: name.to_owned(),
            direction,
        }
    }

    fn policy(id: &str, values: &[f64]) -> PolicyVector {
        PolicyVector {
            policy_id: id.to_owned(),
            values: values.to_vec(),
        }
    }

    #[test]
    fn tradeoffs_remain_on_frontier_without_arbitrary_scalarization() {
        let objectives = vec![
            objective("macro_recall", ObjectiveDirection::Maximize),
            objective("review_slots", ObjectiveDirection::Minimize),
            objective("worst_group_recall", ObjectiveDirection::Maximize),
        ];
        let policies = vec![
            policy("recall_first", &[0.90, 500.0, 0.55]),
            policy("budget_first", &[0.82, 300.0, 0.70]),
            policy("balanced", &[0.86, 350.0, 0.68]),
            policy("dominated", &[0.80, 520.0, 0.50]),
        ];

        assert_eq!(
            pareto_frontier(&objectives, &policies).unwrap(),
            vec![
                "balanced".to_owned(),
                "budget_first".to_owned(),
                "recall_first".to_owned(),
            ]
        );
    }

    #[test]
    fn dominated_policy_is_removed_when_another_is_no_worse_everywhere() {
        let objectives = vec![
            objective("recall", ObjectiveDirection::Maximize),
            objective("runtime_ms", ObjectiveDirection::Minimize),
        ];
        let policies = vec![
            policy("A", &[0.9, 100.0]),
            policy("B", &[0.8, 150.0]),
            policy("C", &[0.95, 250.0]),
        ];
        assert_eq!(
            pareto_frontier(&objectives, &policies).unwrap(),
            vec!["A".to_owned(), "C".to_owned()]
        );
    }

    #[test]
    fn equal_policies_do_not_dominate_each_other() {
        let objectives = vec![objective("recall", ObjectiveDirection::Maximize)];
        let policies = vec![policy("A", &[0.8]), policy("B", &[0.8])];
        assert_eq!(
            pareto_frontier(&objectives, &policies).unwrap(),
            vec!["A".to_owned(), "B".to_owned()]
        );
    }

    #[test]
    fn objective_order_is_explicit_and_not_inferred_from_metric_names() {
        let objectives = vec![
            objective("slots", ObjectiveDirection::Minimize),
            objective("recall", ObjectiveDirection::Maximize),
        ];
        let policies = vec![policy("A", &[100.0, 0.8]), policy("B", &[200.0, 0.9])];
        assert!(dominates(&objectives, &policies[0], &policies[1]));
        assert!(!dominates(&objectives, &policies[1], &policies[0]));
    }

    #[test]
    fn empty_objective_list_fails_closed() {
        assert_eq!(
            pareto_frontier(&[], &[policy("A", &[])]),
            Err(FrontierError::NoObjectives)
        );
    }

    #[test]
    fn duplicate_objective_names_are_rejected() {
        let objectives = vec![
            objective("recall", ObjectiveDirection::Maximize),
            objective("recall", ObjectiveDirection::Minimize),
        ];
        assert_eq!(
            pareto_frontier(&objectives, &[policy("A", &[0.8, 100.0])]),
            Err(FrontierError::DuplicateObjectiveName("recall".to_owned()))
        );
    }

    #[test]
    fn mismatched_vector_length_is_rejected() {
        let objectives = vec![
            objective("recall", ObjectiveDirection::Maximize),
            objective("slots", ObjectiveDirection::Minimize),
        ];
        assert_eq!(
            pareto_frontier(&objectives, &[policy("A", &[0.8])]),
            Err(FrontierError::ObjectiveCountMismatch {
                policy_id: "A".to_owned(),
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn nonfinite_metrics_are_rejected_not_zero_filled() {
        let objectives = vec![objective("recall", ObjectiveDirection::Maximize)];
        assert!(matches!(
            pareto_frontier(&objectives, &[policy("A", &[f64::NAN])]),
            Err(FrontierError::NonFiniteValue { policy_id, objective, value })
                if policy_id == "A" && objective == "recall" && value.is_nan()
        ));
    }

    #[test]
    fn duplicate_policy_ids_are_rejected() {
        let objectives = vec![objective("recall", ObjectiveDirection::Maximize)];
        assert_eq!(
            pareto_frontier(&objectives, &[policy("A", &[0.8]), policy("A", &[0.9])]),
            Err(FrontierError::DuplicatePolicyId("A".to_owned()))
        );
    }

    #[test]
    fn empty_policy_list_has_empty_frontier() {
        let objectives = vec![objective("recall", ObjectiveDirection::Maximize)];
        assert_eq!(pareto_frontier(&objectives, &[]).unwrap(), Vec::<String>::new());
    }
}
