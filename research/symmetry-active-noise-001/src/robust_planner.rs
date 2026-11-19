// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact bounded-outcome adaptive minimax planning.
//!
//! Unlike the deterministic planner, each hypothesis/probe pair may allow
//! several outcomes. The planner enumerates every possible observation branch
//! and optimizes the worst-case surviving hypothesis count, then cumulative
//! cost. It never assumes a probability distribution.

use std::collections::BTreeMap;
use std::fmt;

use symmetry_active_experiment_reference::QualificationBudgetV1;

use super::{BoundedNoiseProblemV1, NoiseErrorV1};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RobustPlannerErrorV1 {
    InvalidProblem(NoiseErrorV1),
    InvalidBudget,
    InvalidHypothesisMask,
    SearchNodeBudgetExceeded { visited: u64, maximum: u64 },
    ArithmeticOverflow,
    ReceiptMismatch,
    InvalidHypothesisIndex,
}

impl fmt::Display for RobustPlannerErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RobustPlannerErrorV1 {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RobustLeafReasonV1 {
    Identified,
    DepthLimit,
    CostLimit,
    NoInformativeProbe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RobustPolicyOutcomeV1 {
    Value(u32),
    /// The probe was attempted but produced no usable observation.
    /// This branch preserves the complete active hypothesis mask.
    Abstained,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobustPolicyBranchV1 {
    pub outcome: RobustPolicyOutcomeV1,
    pub child: Box<RobustPolicyNodeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobustPolicyNodeV1 {
    pub active_hypothesis_mask: u64,
    pub worst_case_remaining: usize,
    pub worst_case_cost: u64,
    pub probe_id: Option<String>,
    pub probe_cost: u64,
    pub leaf_reason: Option<RobustLeafReasonV1>,
    /// Complete sorted set of every outcome allowed by any active hypothesis.
    pub branches: Vec<RobustPolicyBranchV1>,
}

impl RobustPolicyNodeV1 {
    fn leaf(mask: u64, reason: RobustLeafReasonV1) -> Self {
        Self {
            active_hypothesis_mask: mask,
            worst_case_remaining: mask.count_ones() as usize,
            worst_case_cost: 0,
            probe_id: None,
            probe_cost: 0,
            leaf_reason: Some(reason),
            branches: Vec::new(),
        }
    }

    pub fn is_identified(&self) -> bool {
        self.worst_case_remaining == 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RobustPolicySummaryV1 {
    pub worst_case_remaining: usize,
    pub worst_case_cost: u64,
    pub search_nodes_visited: u64,
    pub identified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobustPolicyReceiptV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub problem_contract_bytes: Vec<u8>,
    pub budget: QualificationBudgetV1,
    pub root: RobustPolicyNodeV1,
    pub search_nodes_visited: u64,
}

fn full_mask(count: usize) -> u64 {
    if count == 64 {
        u64::MAX
    } else {
        (1_u64 << count) - 1
    }
}

fn indices(mask: u64, count: usize) -> impl Iterator<Item = usize> {
    (0..count).filter(move |index| mask & (1_u64 << index) != 0)
}

fn outcome_partitions(
    problem: &BoundedNoiseProblemV1,
    active_mask: u64,
    probe_index: usize,
) -> BTreeMap<RobustPolicyOutcomeV1, u64> {
    let mut groups = BTreeMap::new();
    for hypothesis_index in indices(active_mask, problem.hypothesis_ids.len()) {
        for outcome in &problem.allowed_outcomes_by_hypothesis[hypothesis_index][probe_index] {
            *groups
                .entry(RobustPolicyOutcomeV1::Value(*outcome))
                .or_insert(0) |= 1_u64 << hypothesis_index;
        }
    }
    if problem.probes[probe_index].may_abstain {
        // Missingness is a probe-level event, not evidence about which
        // hypothesis generated the observation.
        groups.insert(RobustPolicyOutcomeV1::Abstained, active_mask);
    }
    groups
}

fn solve_node(
    problem: &BoundedNoiseProblemV1,
    active_mask: u64,
    available_probe_mask: u64,
    depth_remaining: u8,
    cost_remaining: u64,
    budget: QualificationBudgetV1,
    visited: &mut u64,
) -> Result<RobustPolicyNodeV1, RobustPlannerErrorV1> {
    *visited = visited
        .checked_add(1)
        .ok_or(RobustPlannerErrorV1::ArithmeticOverflow)?;
    if *visited > budget.max_search_nodes {
        return Err(RobustPlannerErrorV1::SearchNodeBudgetExceeded {
            visited: *visited,
            maximum: budget.max_search_nodes,
        });
    }

    if active_mask.count_ones() == 1 {
        return Ok(RobustPolicyNodeV1::leaf(
            active_mask,
            RobustLeafReasonV1::Identified,
        ));
    }
    if depth_remaining == 0 {
        return Ok(RobustPolicyNodeV1::leaf(
            active_mask,
            RobustLeafReasonV1::DepthLimit,
        ));
    }

    let mut best: Option<RobustPolicyNodeV1> = None;
    let mut had_informative_probe = false;
    let mut had_affordable_probe = false;

    for probe_index in indices(available_probe_mask, problem.probes.len()) {
        let probe = &problem.probes[probe_index];
        if !probe.admissible {
            continue;
        }
        let partitions = outcome_partitions(problem, active_mask, probe_index);
        if partitions.len() <= 1 {
            continue;
        }
        had_informative_probe = true;
        if probe.cost > cost_remaining {
            continue;
        }
        had_affordable_probe = true;

        let next_available = available_probe_mask & !(1_u64 << probe_index);
        let next_cost = cost_remaining - probe.cost;
        let mut branches = Vec::with_capacity(partitions.len());
        let mut worst_case_remaining = 0usize;
        let mut worst_case_child_cost = 0u64;

        for (outcome, child_mask) in partitions {
            let child = solve_node(
                problem,
                child_mask,
                next_available,
                depth_remaining - 1,
                next_cost,
                budget,
                visited,
            )?;
            worst_case_remaining = worst_case_remaining.max(child.worst_case_remaining);
            worst_case_child_cost = worst_case_child_cost.max(child.worst_case_cost);
            branches.push(RobustPolicyBranchV1 {
                outcome,
                child: Box::new(child),
            });
        }

        let worst_case_cost = probe
            .cost
            .checked_add(worst_case_child_cost)
            .ok_or(RobustPlannerErrorV1::ArithmeticOverflow)?;
        let candidate = RobustPolicyNodeV1 {
            active_hypothesis_mask: active_mask,
            worst_case_remaining,
            worst_case_cost,
            probe_id: Some(probe.probe_id.clone()),
            probe_cost: probe.cost,
            leaf_reason: None,
            branches,
        };
        let replace = best.as_ref().is_none_or(|current| {
            (
                candidate.worst_case_remaining,
                candidate.worst_case_cost,
                candidate.probe_id.as_deref().unwrap_or_default(),
            ) < (
                current.worst_case_remaining,
                current.worst_case_cost,
                current.probe_id.as_deref().unwrap_or_default(),
            )
        });
        if replace {
            best = Some(candidate);
        }
    }

    if let Some(best) = best {
        return Ok(best);
    }
    let reason = if had_informative_probe && !had_affordable_probe {
        RobustLeafReasonV1::CostLimit
    } else {
        RobustLeafReasonV1::NoInformativeProbe
    };
    Ok(RobustPolicyNodeV1::leaf(active_mask, reason))
}

/// Exhaustive bounded-outcome minimax policy search.
///
/// Objective order:
/// 1. minimize worst-case surviving hypotheses;
/// 2. minimize worst-case cumulative probe cost;
/// 3. deterministic lexical tie-break on the first probe ID.
///
/// Every allowed outcome receives a branch. A probe can be used at most once,
/// so repeated measurements are not silently treated as independent evidence.
/// Node exhaustion returns an error and no partial policy.
pub fn solve_bounded_noise_policy(
    problem: &BoundedNoiseProblemV1,
    budget: QualificationBudgetV1,
) -> Result<RobustPolicyReceiptV1, RobustPlannerErrorV1> {
    problem
        .validate()
        .map_err(RobustPlannerErrorV1::InvalidProblem)?;
    if budget.max_depth == 0 || budget.max_search_nodes == 0 {
        return Err(RobustPlannerErrorV1::InvalidBudget);
    }
    let active_mask = full_mask(problem.hypothesis_ids.len());
    let available_probe_mask = full_mask(problem.probes.len());
    let mut visited = 0u64;
    let root = solve_node(
        problem,
        active_mask,
        available_probe_mask,
        budget.max_depth,
        budget.max_total_cost,
        budget,
        &mut visited,
    )?;
    Ok(RobustPolicyReceiptV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        problem_contract_bytes: problem.canonical_bytes(),
        budget,
        root,
        search_nodes_visited: visited,
    })
}

/// Recompute the exact minimax tree and compare its complete structure.
pub fn verify_bounded_noise_policy(
    problem: &BoundedNoiseProblemV1,
    receipt: &RobustPolicyReceiptV1,
    expected_budget: QualificationBudgetV1,
) -> Result<RobustPolicySummaryV1, RobustPlannerErrorV1> {
    problem
        .validate()
        .map_err(RobustPlannerErrorV1::InvalidProblem)?;
    if receipt.schema_version != 1
        || receipt.problem_id != problem.problem_id
        || receipt.problem_contract_bytes != problem.canonical_bytes()
        || receipt.budget != expected_budget
    {
        return Err(RobustPlannerErrorV1::ReceiptMismatch);
    }
    let replayed = solve_bounded_noise_policy(problem, expected_budget)?;
    if *receipt != replayed {
        return Err(RobustPlannerErrorV1::ReceiptMismatch);
    }
    Ok(RobustPolicySummaryV1 {
        worst_case_remaining: receipt.root.worst_case_remaining,
        worst_case_cost: receipt.root.worst_case_cost,
        search_nodes_visited: receipt.search_nodes_visited,
        identified: receipt.root.is_identified(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NoiseProbeV1, BoundedNoiseProblemV1};

    fn fixture() -> BoundedNoiseProblemV1 {
        BoundedNoiseProblemV1 {
            problem_id: "robust-bounded-adaptive-three-hypothesis-v1".to_owned(),
            hypothesis_ids: vec!["H-A".to_owned(), "H-B".to_owned(), "H-C".to_owned()],
            probes: vec![
                NoiseProbeV1 { probe_id: "p0".to_owned(), cost: 1, admissible: true },
                NoiseProbeV1 { probe_id: "p1".to_owned(), cost: 1, admissible: true },
                NoiseProbeV1 { probe_id: "p2".to_owned(), cost: 1, admissible: true },
            ],
            allowed_outcomes_by_hypothesis: vec![
                vec![vec![0, 1], vec![0], vec![8]],
                vec![vec![1, 2], vec![1], vec![8]],
                vec![vec![2], vec![0, 1], vec![9]],
            ],
        }
    }

    fn assert_all_predictions_have_branches(
        problem: &BoundedNoiseProblemV1,
        node: &RobustPolicyNodeV1,
    ) {
        let Some(probe_id) = node.probe_id.as_ref() else {
            assert!(node.branches.is_empty());
            return;
        };
        let probe_index = problem
            .probes
            .iter()
            .position(|probe| &probe.probe_id == probe_id)
            .unwrap();
        let branch_outcomes: Vec<RobustPolicyOutcomeV1> =
            node.branches.iter().map(|branch| branch.outcome).collect();
        let expected: Vec<RobustPolicyOutcomeV1> =
            outcome_partitions(problem, node.active_hypothesis_mask, probe_index)
                .keys()
                .copied()
                .collect();
        assert_eq!(branch_outcomes, expected);
        for branch in &node.branches {
            for hypothesis_index in indices(
                node.active_hypothesis_mask,
                problem.hypothesis_ids.len(),
            ) {
                if problem.allowed_outcomes_by_hypothesis[hypothesis_index][probe_index]
                    .contains(&branch.outcome)
                {
                    assert_ne!(
                        branch.child.active_hypothesis_mask & (1_u64 << hypothesis_index),
                        0,
                        "the planner eliminated a hypothesis that allows this outcome"
                    );
                }
            }
            assert_all_predictions_have_branches(problem, &branch.child);
        }
    }

    #[test]
    fn robust_policy_covers_every_allowed_outcome_and_identifies_worst_case() {
        let problem = fixture();
        let budget = QualificationBudgetV1::default();
        let receipt = solve_bounded_noise_policy(&problem, budget).unwrap();
        assert_eq!(receipt.root.worst_case_remaining, 1);
        assert_eq!(receipt.root.worst_case_cost, 2);
        assert_all_predictions_have_branches(&problem, &receipt.root);
        let summary = verify_bounded_noise_policy(&problem, &receipt, budget).unwrap();
        assert!(summary.identified);
        assert_eq!(summary.worst_case_cost, 2);
    }

    #[test]
    fn possible_abstention_has_its_own_branch_and_preserves_every_candidate() {
        let mut problem = fixture();
        problem.probes.truncate(1);
        problem.probes[0].may_abstain = true;
        let budget = QualificationBudgetV1::default();
        let receipt = solve_bounded_noise_policy(&problem, budget).unwrap();

        let abstention = receipt
            .root
            .branches
            .iter()
            .find(|branch| branch.outcome == RobustPolicyOutcomeV1::Abstained)
            .expect("declared abstention must have an explicit policy branch");
        assert_eq!(abstention.child.active_hypothesis_mask, full_mask(3));
        assert_eq!(abstention.child.worst_case_remaining, 3);
        assert_eq!(receipt.root.worst_case_remaining, 3);
        assert_all_predictions_have_branches(&problem, &receipt.root);
        verify_bounded_noise_policy(&problem, &receipt, budget).unwrap();
    }

    #[test]
    fn zero_noise_singletons_match_the_deterministic_policy_objective() {
        use symmetry_active_experiment_reference::{
            ActiveExperimentProblemV1, ExperimentProbeV1, compile_active_experiment,
        };

        let deterministic = ActiveExperimentProblemV1 {
            problem_id: "zero-noise-robust-planner-parity-v1".to_owned(),
            hypotheses: vec!["H-A".to_owned(), "H-B".to_owned(), "H-C".to_owned()],
            probes: vec![
                ExperimentProbeV1 {
                    probe_id: "p0".to_owned(),
                    cost: 1,
                    admissible: true,
                    outcomes: vec![0, 1, 2],
                },
                ExperimentProbeV1 {
                    probe_id: "p1".to_owned(),
                    cost: 1,
                    admissible: true,
                    outcomes: vec![0, 1, 0],
                },
                ExperimentProbeV1 {
                    probe_id: "p2".to_owned(),
                    cost: 1,
                    admissible: true,
                    outcomes: vec![8, 8, 9],
                },
            ],
        };
        let bounded = BoundedNoiseProblemV1::from_deterministic(&deterministic).unwrap();
        let budget = QualificationBudgetV1::default();
        let robust = solve_bounded_noise_policy(&bounded, budget).unwrap();
        let exact = compile_active_experiment(&deterministic, budget).unwrap();
        assert_eq!(robust.root.worst_case_remaining, exact.root.worst_case_remaining);
        assert_eq!(robust.root.worst_case_cost, exact.root.worst_case_cost);
    }

    #[test]
    fn ambiguous_probe_family_remains_unresolved_instead_of_fabricating_identification() {
        let mut problem = fixture();
        problem.probes.truncate(1);
        problem.allowed_outcomes_by_hypothesis = vec![
            vec![vec![0, 1]],
            vec![vec![1, 2]],
            vec![vec![0, 2]],
        ];
        let receipt = solve_bounded_noise_policy(&problem, QualificationBudgetV1::default()).unwrap();
        assert_eq!(receipt.root.worst_case_remaining, 2);
        assert_eq!(receipt.root.worst_case_cost, 1);
        assert!(!receipt.root.is_identified());
    }

    #[test]
    fn node_depth_and_cost_budgets_fail_closed_or_preserve_the_frontier() {
        let problem = fixture();
        assert!(matches!(
            solve_bounded_noise_policy(
                &problem,
                QualificationBudgetV1 {
                    max_search_nodes: 1,
                    ..QualificationBudgetV1::default()
                }
            ),
            Err(RobustPlannerErrorV1::SearchNodeBudgetExceeded { .. })
        ));

        let shallow = solve_bounded_noise_policy(
            &problem,
            QualificationBudgetV1 {
                max_depth: 1,
                ..QualificationBudgetV1::default()
            },
        )
        .unwrap();
        assert!(shallow.root.worst_case_remaining > 1);

        let no_cost = solve_bounded_noise_policy(
            &problem,
            QualificationBudgetV1 {
                max_total_cost: 0,
                ..QualificationBudgetV1::default()
            },
        )
        .unwrap();
        assert_eq!(no_cost.root.worst_case_remaining, 3);
        assert_eq!(no_cost.root.leaf_reason, Some(RobustLeafReasonV1::CostLimit));
    }

    #[test]
    fn forged_or_mutated_policy_receipts_are_rejected() {
        let problem = fixture();
        let receipt = solve_bounded_noise_policy(&problem, QualificationBudgetV1::default()).unwrap();
        verify_bounded_noise_policy(&problem, &receipt, QualificationBudgetV1::default()).unwrap();

        let mut forged = receipt.clone();
        forged.root.worst_case_remaining = 3;
        assert_eq!(
            verify_bounded_noise_policy(&problem, &forged, QualificationBudgetV1::default()),
            Err(RobustPlannerErrorV1::ReceiptMismatch)
        );

        let mut changed_problem = problem.clone();
        changed_problem.allowed_outcomes_by_hypothesis[0][0].push(99);
        assert_eq!(
            verify_bounded_noise_policy(&changed_problem, &receipt, QualificationBudgetV1::default()),
            Err(RobustPlannerErrorV1::ReceiptMismatch)
        );

        let changed_budget = QualificationBudgetV1 {
            max_total_cost: QualificationBudgetV1::default().max_total_cost - 1,
            ..QualificationBudgetV1::default()
        };
        assert_eq!(
            verify_bounded_noise_policy(&problem, &receipt, changed_budget),
            Err(RobustPlannerErrorV1::ReceiptMismatch)
        );
    }
}
