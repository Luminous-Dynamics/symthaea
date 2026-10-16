//! Conservative interval-aware Pareto analysis for CRISPR benchmark policy reports.
//!
//! This module does not derive confidence intervals. It accepts preregistered objective
//! intervals supplied by a separate, validated metric layer, then distinguishes:
//! - definite/robust dominance: A is no worse than B even at A's worst and B's best
//!   objective bounds, and is strictly better on at least one objective;
//! - possible dominance: A could dominate B for some independent values within their
//!   intervals, but robust dominance is not established;
//! - unresolved policy pairs: no strict possible dominance in either direction.
//!
//! Bounds are compared coordinate-wise and conservatively; they do not encode covariance,
//! paired resampling, multiplicity correction, or joint coverage. The caller must document
//! how every interval was produced and should use simultaneous/group-aware bounds when a
//! joint claim over multiple objectives/policies is needed. Never interpret this output as
//! biological safety, statistical significance, or proof of model improvement.
//!
//! Compile and run without Cargo/network:
//!   rustc --edition=2024 --test scripts/crispr_interval_frontier.rs -o /tmp/crispr-interval-tests
//!   /tmp/crispr-interval-tests

use std::collections::HashSet;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectiveDirection {
    Maximize,
    Minimize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntervalObjective {
    pub name: String,
    pub direction: ObjectiveDirection,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricInterval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyIntervalVector {
    pub policy_id: String,
    /// Each interval is in the objective's original units and direction.
    pub values: Vec<MetricInterval>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IntervalFrontierError {
    NoObjectives,
    DuplicateObjectiveName(String),
    DuplicatePolicyId(String),
    ObjectiveCountMismatch {
        policy_id: String,
        expected: usize,
        actual: usize,
    },
    InvalidInterval {
        policy_id: String,
        objective: String,
        reason: String,
    },
}

impl fmt::Display for IntervalFrontierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoObjectives => write!(f, "at least one objective is required"),
            Self::DuplicateObjectiveName(name) => write!(f, "duplicate objective name: {name}"),
            Self::DuplicatePolicyId(id) => write!(f, "duplicate policy id: {id}"),
            Self::ObjectiveCountMismatch {
                policy_id,
                expected,
                actual,
            } => write!(
                f,
                "policy {policy_id} has {actual} objectives; expected {expected}"
            ),
            Self::InvalidInterval {
                policy_id,
                objective,
                reason,
            } => write!(
                f,
                "policy {policy_id} has invalid interval for {objective}: {reason}"
            ),
        }
    }
}

impl std::error::Error for IntervalFrontierError {}

#[derive(Clone, Debug, PartialEq)]
pub struct IntervalFrontier {
    /// Policies not definitely dominated by another policy. Overlapping intervals
    /// deliberately keep a policy on this conservative frontier.
    pub robust_frontier_ids: Vec<String>,
    /// Policies definitely dominated by at least one competitor under the interval bounds.
    pub definitely_dominated_ids: Vec<String>,
    /// (dominator, dominated) pairs whose dominance is true for all interval values.
    pub robust_dominator_pairs: Vec<(String, String)>,
    /// (potential dominator, potential target) pairs where dominance is feasible within
    /// the intervals but not established for all values.
    pub possible_only_dominator_pairs: Vec<(String, String)>,
}

fn validate(
    objectives: &[IntervalObjective],
    policies: &[PolicyIntervalVector],
) -> Result<(), IntervalFrontierError> {
    if objectives.is_empty() {
        return Err(IntervalFrontierError::NoObjectives);
    }
    let mut names = HashSet::with_capacity(objectives.len());
    for objective in objectives {
        if objective.name.trim().is_empty() || !names.insert(objective.name.as_str()) {
            return Err(IntervalFrontierError::DuplicateObjectiveName(
                objective.name.clone(),
            ));
        }
    }

    let mut ids = HashSet::with_capacity(policies.len());
    for policy in policies {
        if policy.policy_id.trim().is_empty() || !ids.insert(policy.policy_id.as_str()) {
            return Err(IntervalFrontierError::DuplicatePolicyId(
                policy.policy_id.clone(),
            ));
        }
        if policy.values.len() != objectives.len() {
            return Err(IntervalFrontierError::ObjectiveCountMismatch {
                policy_id: policy.policy_id.clone(),
                expected: objectives.len(),
                actual: policy.values.len(),
            });
        }
        for (objective, interval) in objectives.iter().zip(&policy.values) {
            if !interval.lower.is_finite() || !interval.upper.is_finite() {
                return Err(IntervalFrontierError::InvalidInterval {
                    policy_id: policy.policy_id.clone(),
                    objective: objective.name.clone(),
                    reason: "both bounds must be finite".to_owned(),
                });
            }
            if interval.lower > interval.upper {
                return Err(IntervalFrontierError::InvalidInterval {
                    policy_id: policy.policy_id.clone(),
                    objective: objective.name.clone(),
                    reason: format!(
                        "lower bound {} exceeds upper bound {}",
                        interval.lower, interval.upper
                    ),
                });
            }
        }
    }
    Ok(())
}

/// True only if A is no worse than B for every possible realization in each declared
/// objective interval and strictly better for at least one objective.
pub fn robustly_dominates(
    objectives: &[IntervalObjective],
    a: &PolicyIntervalVector,
    b: &PolicyIntervalVector,
) -> bool {
    let mut strict = false;
    for (objective, (av, bv)) in objectives.iter().zip(a.values.iter().zip(&b.values)) {
        match objective.direction {
            ObjectiveDirection::Maximize => {
                if av.lower < bv.upper {
                    return false;
                }
                strict |= av.lower > bv.upper;
            }
            ObjectiveDirection::Minimize => {
                if av.upper > bv.lower {
                    return false;
                }
                strict |= av.upper < bv.lower;
            }
        }
    }
    strict
}

/// True if A could dominate B for at least one combination of values within the declared
/// intervals, under coordinate-wise independent interval uncertainty. This is only a
/// possibility flag; overlapping objective intervals may be correlated in ways this module
/// cannot express.
pub fn possibly_dominates(
    objectives: &[IntervalObjective],
    a: &PolicyIntervalVector,
    b: &PolicyIntervalVector,
) -> bool {
    let mut strict_possible = false;
    for (objective, (av, bv)) in objectives.iter().zip(a.values.iter().zip(&b.values)) {
        match objective.direction {
            ObjectiveDirection::Maximize => {
                if av.upper < bv.lower {
                    return false;
                }
                strict_possible |= av.upper > bv.lower;
            }
            ObjectiveDirection::Minimize => {
                if av.lower > bv.upper {
                    return false;
                }
                strict_possible |= av.lower < bv.upper;
            }
        }
    }
    strict_possible
}

pub fn analyse_interval_frontier(
    objectives: &[IntervalObjective],
    policies: &[PolicyIntervalVector],
) -> Result<IntervalFrontier, IntervalFrontierError> {
    validate(objectives, policies)?;
    let mut robust_dominator_pairs = Vec::new();
    let mut possible_only_dominator_pairs = Vec::new();
    let mut robust_dominated = HashSet::new();

    for i in 0..policies.len() {
        for j in 0..policies.len() {
            if i == j {
                continue;
            }
            let a = &policies[i];
            let b = &policies[j];
            if robustly_dominates(objectives, a, b) {
                robust_dominated.insert(b.policy_id.as_str());
                robust_dominator_pairs.push((a.policy_id.clone(), b.policy_id.clone()));
            } else if possibly_dominates(objectives, a, b) {
                possible_only_dominator_pairs.push((a.policy_id.clone(), b.policy_id.clone()));
            }
        }
    }

    robust_dominator_pairs.sort();
    possible_only_dominator_pairs.sort();
    let mut robust_frontier_ids: Vec<String> = policies
        .iter()
        .filter(|p| !robust_dominated.contains(p.policy_id.as_str()))
        .map(|p| p.policy_id.clone())
        .collect();
    robust_frontier_ids.sort();

    let mut definitely_dominated_ids: Vec<String> =
        robust_dominated.into_iter().map(str::to_owned).collect();
    definitely_dominated_ids.sort();

    Ok(IntervalFrontier {
        robust_frontier_ids,
        definitely_dominated_ids,
        robust_dominator_pairs,
        possible_only_dominator_pairs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn objective(name: &str, direction: ObjectiveDirection) -> IntervalObjective {
        IntervalObjective { name: name.to_owned(), direction }
    }

    fn policy(id: &str, values: &[(f64, f64)]) -> PolicyIntervalVector {
        PolicyIntervalVector {
            policy_id: id.to_owned(),
            values: values.iter().map(|(lower, upper)| MetricInterval { lower: *lower, upper: *upper }).collect(),
        }
    }

    #[test]
    fn separated_maximize_intervals_yield_robust_dominance() {
        let objectives = vec![objective("recall", ObjectiveDirection::Maximize)];
        let policies = vec![policy("A", &[(0.9, 0.95)]), policy("B", &[(0.7, 0.8)])];
        let result = analyse_interval_frontier(&objectives, &policies).unwrap();
        assert_eq!(result.robust_frontier_ids, vec!["A"]);
        assert_eq!(result.definitely_dominated_ids, vec!["B"]);
        assert_eq!(result.robust_dominator_pairs, vec![("A".to_owned(), "B".to_owned())]);
        assert!(result.possible_only_dominator_pairs.is_empty());
    }

    #[test]
    fn separated_minimize_intervals_yield_robust_dominance() {
        let objectives = vec![objective("risk", ObjectiveDirection::Minimize)];
        let policies = vec![policy("A", &[(0.01, 0.02)]), policy("B", &[(0.04, 0.06)])];
        let result = analyse_interval_frontier(&objectives, &policies).unwrap();
        assert_eq!(result.robust_frontier_ids, vec!["A"]);
        assert_eq!(result.definitely_dominated_ids, vec!["B"]);
    }

    #[test]
    fn overlapping_intervals_are_possible_not_robust_dominance() {
        let objectives = vec![objective("recall", ObjectiveDirection::Maximize)];
        let policies = vec![policy("A", &[(0.7, 0.9)]), policy("B", &[(0.8, 0.95)])];
        let result = analyse_interval_frontier(&objectives, &policies).unwrap();
        assert_eq!(result.robust_frontier_ids, vec!["A", "B"]);
        assert!(result.definitely_dominated_ids.is_empty());
        assert_eq!(
            result.possible_only_dominator_pairs,
            vec![("A".to_owned(), "B".to_owned()), ("B".to_owned(), "A".to_owned())]
        );
    }

    #[test]
    fn equal_point_intervals_do_not_strictly_dominate() {
        let objectives = vec![objective("recall", ObjectiveDirection::Maximize)];
        let policies = vec![policy("A", &[(0.8, 0.8)]), policy("B", &[(0.8, 0.8)])];
        let result = analyse_interval_frontier(&objectives, &policies).unwrap();
        assert_eq!(result.robust_frontier_ids, vec!["A", "B"]);
        assert!(result.robust_dominator_pairs.is_empty());
        assert!(result.possible_only_dominator_pairs.is_empty());
    }

    #[test]
    fn strict_tradeoff_is_incomparable_in_both_directions() {
        let objectives = vec![
            objective("recall", ObjectiveDirection::Maximize),
            objective("slots", ObjectiveDirection::Minimize),
        ];
        let policies = vec![policy("A", &[(0.9, 0.9), (200.0, 200.0)]), policy("B", &[(0.8, 0.8), (100.0, 100.0)])];
        let result = analyse_interval_frontier(&objectives, &policies).unwrap();
        assert_eq!(result.robust_frontier_ids, vec!["A", "B"]);
        assert!(result.robust_dominator_pairs.is_empty());
        assert!(result.possible_only_dominator_pairs.is_empty());
    }

    #[test]
    fn possible_dominance_on_one_coordinate_is_not_a_frontier_removal() {
        let objectives = vec![
            objective("recall", ObjectiveDirection::Maximize),
            objective("slots", ObjectiveDirection::Minimize),
        ];
        let policies = vec![policy("A", &[(0.8, 0.9), (200.0, 250.0)]), policy("B", &[(0.7, 0.85), (230.0, 300.0)])];
        let result = analyse_interval_frontier(&objectives, &policies).unwrap();
        assert_eq!(result.robust_frontier_ids, vec!["A", "B"]);
        assert_eq!(result.possible_only_dominator_pairs, vec![("A".to_owned(), "B".to_owned())]);
    }

    #[test]
    fn empty_policy_list_has_empty_frontier() {
        let objectives = vec![objective("recall", ObjectiveDirection::Maximize)];
        let result = analyse_interval_frontier(&objectives, &[]).unwrap();
        assert!(result.robust_frontier_ids.is_empty());
        assert!(result.definitely_dominated_ids.is_empty());
    }

    #[test]
    fn empty_objective_list_fails_closed() {
        assert_eq!(
            analyse_interval_frontier(&[], &[]),
            Err(IntervalFrontierError::NoObjectives)
        );
    }

    #[test]
    fn duplicate_objective_names_fail_closed() {
        let objectives = vec![
            objective("recall", ObjectiveDirection::Maximize),
            objective("recall", ObjectiveDirection::Minimize),
        ];
        assert_eq!(
            analyse_interval_frontier(&objectives, &[policy("A", &[(0.7, 0.8), (1.0, 2.0)])]),
            Err(IntervalFrontierError::DuplicateObjectiveName("recall".to_owned()))
        );
    }

    #[test]
    fn duplicate_policy_ids_fail_closed() {
        let objectives = vec![objective("recall", ObjectiveDirection::Maximize)];
        assert_eq!(
            analyse_interval_frontier(&objectives, &[policy("A", &[(0.7, 0.8)]), policy("A", &[(0.8, 0.9)])]),
            Err(IntervalFrontierError::DuplicatePolicyId("A".to_owned()))
        );
    }

    #[test]
    fn objective_dimension_mismatch_fails_closed() {
        let objectives = vec![objective("recall", ObjectiveDirection::Maximize), objective("slots", ObjectiveDirection::Minimize)];
        assert_eq!(
            analyse_interval_frontier(&objectives, &[policy("A", &[(0.8, 0.9)])]),
            Err(IntervalFrontierError::ObjectiveCountMismatch { policy_id: "A".to_owned(), expected: 2, actual: 1 })
        );
    }

    #[test]
    fn reversed_interval_fails_closed() {
        let objectives = vec![objective("recall", ObjectiveDirection::Maximize)];
        assert!(matches!(
            analyse_interval_frontier(&objectives, &[policy("A", &[(0.9, 0.8)])]),
            Err(IntervalFrontierError::InvalidInterval { .. })
        ));
    }

    #[test]
    fn non_finite_interval_fails_closed() {
        let objectives = vec![objective("recall", ObjectiveDirection::Maximize)];
        assert!(matches!(
            analyse_interval_frontier(&objectives, &[policy("A", &[(0.8, f64::INFINITY)])]),
            Err(IntervalFrontierError::InvalidInterval { .. })
        ));
    }

    #[test]
    fn duplicate_interval_values_can_keep_both_policies() {
        let objectives = vec![objective("recall", ObjectiveDirection::Maximize), objective("slots", ObjectiveDirection::Minimize)];
        let policies = vec![policy("A", &[(0.8, 0.9), (100.0, 200.0)]), policy("B", &[(0.8, 0.9), (100.0, 200.0)])];
        let result = analyse_interval_frontier(&objectives, &policies).unwrap();
        assert_eq!(result.robust_frontier_ids, vec!["A", "B"]);
        assert!(result.robust_dominator_pairs.is_empty());
    }

    #[test]
    fn exact_strict_boundary_counts_as_robust_dominance() {
        let objectives = vec![objective("recall", ObjectiveDirection::Maximize)];
        let policies = vec![policy("A", &[(0.8, 0.9)]), policy("B", &[(0.7, 0.8)])];
        let result = analyse_interval_frontier(&objectives, &policies).unwrap();
        assert_eq!(result.definitely_dominated_ids, vec!["B"]);
    }
}
