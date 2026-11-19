// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact adaptive minimax planning with campaign-persistent latent state.
//!
//! Each hypothesis carries a set of still-possible persistent error states.
//! Every observation branch intersects that set with the states predicting the
//! observed outcome. A path can narrow a hypothesis's latent state, but it can
//! never switch to a new state between probes.
//!
//! The objective is exact finite minimax, not a posterior or probability model.

use std::collections::BTreeMap;
use std::fmt;

use symmetry_active_experiment_reference::QualificationBudgetV1;

use super::{NoiseErrorV1, PersistentNoiseProblemV1};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistentPlannerErrorV1 {
    InvalidProblem(NoiseErrorV1),
    InvalidBudget,
    InvalidStateMasks,
    SearchNodeBudgetExceeded { visited: u64, maximum: u64 },
    ArithmeticOverflow,
    ReceiptMismatch,
}

impl fmt::Display for PersistentPlannerErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PersistentPlannerErrorV1 {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistentLeafReasonV1 {
    Identified,
    DepthLimit,
    CostLimit,
    NoInformativeProbe,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentPolicyBranchV1 {
    pub outcome: u32,
    pub child: Box<PersistentPolicyNodeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentPolicyNodeV1 {
    /// Bit i is set iff hypothesis i remains compatible on this path.
    pub active_hypothesis_mask: u64,
    /// Per-hypothesis latent-state masks. These are refined, never reset,
    /// along a path and are the core persistent-noise invariant.
    pub active_error_state_masks: Vec<u64>,
    pub worst_case_remaining: usize,
    pub worst_case_cost: u64,
    pub probe_id: Option<String>,
    pub probe_cost: u64,
    pub leaf_reason: Option<PersistentLeafReasonV1>,
    /// Complete sorted set of every outcome allowed by any active (hypothesis,
    /// persistent-state) pair.
    pub branches: Vec<PersistentPolicyBranchV1>,
}

impl PersistentPolicyNodeV1 {
    fn leaf(
        state_masks: Vec<u64>,
        reason: PersistentLeafReasonV1,
    ) -> Self {
        let active_hypothesis_mask = mask_from_states(&state_masks);
        Self {
            active_hypothesis_mask,
            worst_case_remaining: active_hypothesis_mask.count_ones() as usize,
            active_error_state_masks: state_masks,
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
pub struct PersistentPolicySummaryV1 {
    pub worst_case_remaining: usize,
    pub worst_case_cost: u64,
    pub search_nodes_visited: u64,
    pub identified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentPolicyReceiptV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub problem_contract_bytes: Vec<u8>,
    pub budget: QualificationBudgetV1,
    pub root: PersistentPolicyNodeV1,
    pub search_nodes_visited: u64,
}

fn full_mask(count: usize) -> u64 {
    if count == 64 {
        u64::MAX
    } else {
        (1_u64 << count) - 1
    }
}

fn mask_from_states(state_masks: &[u64]) -> u64 {
    state_masks.iter().enumerate().fold(0_u64, |mask, (index, states)| {
        if *states == 0 {
            mask
        } else {
            mask | (1_u64 << index)
        }
    })
}

fn indices(mask: u64, count: usize) -> impl Iterator<Item = usize> {
    (0..count).filter(move |index| mask & (1_u64 << index) != 0)
}

fn validate_state_masks(
    problem: &PersistentNoiseProblemV1,
    state_masks: &[u64],
) -> Result<(), PersistentPlannerErrorV1> {
    if state_masks.len() != problem.hypothesis_ids.len() {
        return Err(PersistentPlannerErrorV1::InvalidStateMasks);
    }
    let valid_states = full_mask(problem.error_state_ids.len());
    if state_masks.iter().any(|mask| mask & !valid_states != 0) {
        return Err(PersistentPlannerErrorV1::InvalidStateMasks);
    }
    if state_masks.iter().all(|mask| *mask == 0) {
        return Err(PersistentPlannerErrorV1::InvalidStateMasks);
    }
    Ok(())
}

/// For every outcome possible from an active (hypothesis, fixed-state) pair,
/// build the child masks by intersecting the parent's state masks with states
/// that predict that outcome for this probe.
fn outcome_partitions(
    problem: &PersistentNoiseProblemV1,
    state_masks: &[u64],
    probe_index: usize,
) -> BTreeMap<u32, Vec<u64>> {
    let mut groups: BTreeMap<u32, Vec<u64>> = BTreeMap::new();
    for (hypothesis_index, active_states) in state_masks.iter().copied().enumerate() {
        for state_index in indices(active_states, problem.error_state_ids.len()) {
            let outcome =
                problem.predicted_outcomes_by_hypothesis[hypothesis_index][probe_index]
                    [state_index];
            let child_masks = groups
                .entry(outcome)
                .or_insert_with(|| vec![0_u64; problem.hypothesis_ids.len()]);
            child_masks[hypothesis_index] |= 1_u64 << state_index;
        }
    }
    groups
}

fn solve_node(
    problem: &PersistentNoiseProblemV1,
    state_masks: &[u64],
    available_probe_mask: u64,
    depth_remaining: u8,
    cost_remaining: u64,
    budget: QualificationBudgetV1,
    visited: &mut u64,
) -> Result<PersistentPolicyNodeV1, PersistentPlannerErrorV1> {
    *visited = visited
        .checked_add(1)
        .ok_or(PersistentPlannerErrorV1::ArithmeticOverflow)?;
    if *visited > budget.max_search_nodes {
        return Err(PersistentPlannerErrorV1::SearchNodeBudgetExceeded {
            visited: *visited,
            maximum: budget.max_search_nodes,
        });
    }
    validate_state_masks(problem, state_masks)?;

    if mask_from_states(state_masks).count_ones() == 1 {
        return Ok(PersistentPolicyNodeV1::leaf(
            state_masks.to_vec(),
            PersistentLeafReasonV1::Identified,
        ));
    }
    if depth_remaining == 0 {
        return Ok(PersistentPolicyNodeV1::leaf(
            state_masks.to_vec(),
            PersistentLeafReasonV1::DepthLimit,
        ));
    }

    let mut best: Option<PersistentPolicyNodeV1> = None;
    let mut had_informative_probe = false;
    let mut had_affordable_probe = false;

    for probe_index in indices(available_probe_mask, problem.probes.len()) {
        let probe = &problem.probes[probe_index];
        if !probe.admissible {
            continue;
        }
        let partitions = outcome_partitions(problem, state_masks, probe_index);
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
        let mut worst_case_remaining = 0_usize;
        let mut worst_case_child_cost = 0_u64;

        for (outcome, child_state_masks) in partitions {
            let child = solve_node(
                problem,
                &child_state_masks,
                next_available,
                depth_remaining - 1,
                next_cost,
                budget,
                visited,
            )?;
            worst_case_remaining = worst_case_remaining.max(child.worst_case_remaining);
            worst_case_child_cost = worst_case_child_cost.max(child.worst_case_cost);
            branches.push(PersistentPolicyBranchV1 {
                outcome,
                child: Box::new(child),
            });
        }

        let worst_case_cost = probe
            .cost
            .checked_add(worst_case_child_cost)
            .ok_or(PersistentPlannerErrorV1::ArithmeticOverflow)?;
        let candidate = PersistentPolicyNodeV1 {
            active_hypothesis_mask: mask_from_states(state_masks),
            active_error_state_masks: state_masks.to_vec(),
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
        PersistentLeafReasonV1::CostLimit
    } else {
        PersistentLeafReasonV1::NoInformativeProbe
    };
    Ok(PersistentPolicyNodeV1::leaf(state_masks.to_vec(), reason))
}

/// Exhaustive adaptive minimax search under one fixed but initially unknown
/// latent state per candidate hypothesis. Every possible result is branched;
/// each branch only removes incompatible states. A candidate cannot adopt a
/// different latent state at a later step.
///
/// Objective order: worst-case surviving hypotheses, worst-case cumulative
/// cost, then deterministic lexical first-probe tie-break.
pub fn solve_persistent_noise_policy(
    problem: &PersistentNoiseProblemV1,
    budget: QualificationBudgetV1,
) -> Result<PersistentPolicyReceiptV1, PersistentPlannerErrorV1> {
    problem
        .validate()
        .map_err(PersistentPlannerErrorV1::InvalidProblem)?;
    if budget.max_depth == 0 || budget.max_search_nodes == 0 {
        return Err(PersistentPlannerErrorV1::InvalidBudget);
    }

    let state_masks = vec![full_mask(problem.error_state_ids.len()); problem.hypothesis_ids.len()];
    let mut visited = 0_u64;
    let root = solve_node(
        problem,
        &state_masks,
        full_mask(problem.probes.len()),
        budget.max_depth,
        budget.max_total_cost,
        budget,
        &mut visited,
    )?;
    Ok(PersistentPolicyReceiptV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        problem_contract_bytes: problem.canonical_bytes(),
        budget,
        root,
        search_nodes_visited: visited,
    })
}

/// Recompute the exact policy. The expected budget is supplied independently;
/// a receipt cannot choose the resource envelope against which it is checked.
pub fn verify_persistent_noise_policy(
    problem: &PersistentNoiseProblemV1,
    receipt: &PersistentPolicyReceiptV1,
    expected_budget: QualificationBudgetV1,
) -> Result<PersistentPolicySummaryV1, PersistentPlannerErrorV1> {
    problem
        .validate()
        .map_err(PersistentPlannerErrorV1::InvalidProblem)?;
    if receipt.schema_version != 1
        || receipt.problem_id != problem.problem_id
        || receipt.problem_contract_bytes != problem.canonical_bytes()
        || receipt.budget != expected_budget
    {
        return Err(PersistentPlannerErrorV1::ReceiptMismatch);
    }
    let replayed = solve_persistent_noise_policy(problem, expected_budget)?;
    if *receipt != replayed {
        return Err(PersistentPlannerErrorV1::ReceiptMismatch);
    }
    Ok(PersistentPolicySummaryV1 {
        worst_case_remaining: receipt.root.worst_case_remaining,
        worst_case_cost: receipt.root.worst_case_cost,
        search_nodes_visited: receipt.search_nodes_visited,
        identified: receipt.root.is_identified(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NoiseProbeV1;

    fn resolvable_fixture() -> PersistentNoiseProblemV1 {
        PersistentNoiseProblemV1 {
            problem_id: "persistent-adaptive-two-hypothesis-v1".to_owned(),
            hypothesis_ids: vec!["H-A".to_owned(), "H-B".to_owned()],
            probes: vec![
                NoiseProbeV1 {
                    probe_id: "p-left".to_owned(),
                    cost: 1,
                    admissible: true,
                },
                NoiseProbeV1 {
                    probe_id: "p-right".to_owned(),
                    cost: 1,
                    admissible: true,
                },
            ],
            error_state_ids: vec!["low".to_owned(), "high".to_owned()],
            predicted_outcomes_by_hypothesis: vec![
                vec![vec![0, 1], vec![0, 1]],
                vec![vec![0, 1], vec![1, 0]],
            ],
        }
    }

    fn cross_state_collision_fixture() -> PersistentNoiseProblemV1 {
        PersistentNoiseProblemV1 {
            problem_id: "persistent-adaptive-cross-state-collision-v1".to_owned(),
            hypothesis_ids: vec!["H-A".to_owned(), "H-B".to_owned()],
            probes: vec![
                NoiseProbeV1 {
                    probe_id: "p-left".to_owned(),
                    cost: 1,
                    admissible: true,
                },
                NoiseProbeV1 {
                    probe_id: "p-right".to_owned(),
                    cost: 1,
                    admissible: true,
                },
            ],
            error_state_ids: vec!["low".to_owned(), "high".to_owned()],
            predicted_outcomes_by_hypothesis: vec![
                vec![vec![0, 1], vec![0, 1]],
                vec![vec![1, 0], vec![1, 0]],
            ],
        }
    }

    fn assert_every_persistent_state_has_a_complete_branch(
        problem: &PersistentNoiseProblemV1,
        node: &PersistentPolicyNodeV1,
    ) {
        let Some(probe_id) = node.probe_id.as_ref() else {
            assert!(node.branches.is_empty());
            assert_eq!(
                node.active_hypothesis_mask,
                mask_from_states(&node.active_error_state_masks)
            );
            return;
        };
        let probe_index = problem
            .probes
            .iter()
            .position(|probe| &probe.probe_id == probe_id)
            .unwrap();
        assert_eq!(
            node.active_hypothesis_mask,
            mask_from_states(&node.active_error_state_masks)
        );
        let expected = outcome_partitions(
            problem,
            &node.active_error_state_masks,
            probe_index,
        );
        assert_eq!(
            node.branches.iter().map(|branch| branch.outcome).collect::<Vec<_>>(),
            expected.keys().copied().collect::<Vec<_>>()
        );
        for branch in &node.branches {
            for hypothesis_index in 0..problem.hypothesis_ids.len() {
                for state_index in indices(
                    node.active_error_state_masks[hypothesis_index],
                    problem.error_state_ids.len(),
                ) {
                    let predicted = problem.predicted_outcomes_by_hypothesis[hypothesis_index]
                        [probe_index][state_index];
                    if predicted == branch.outcome {
                        assert_ne!(
                            branch.child.active_error_state_masks[hypothesis_index]
                                & (1_u64 << state_index),
                            0,
                            "a persistent state compatible with this outcome was eliminated"
                        );
                    } else {
                        assert_eq!(
                            branch.child.active_error_state_masks[hypothesis_index]
                                & (1_u64 << state_index),
                            0,
                            "a state predicting a different outcome survived"
                        );
                    }
                }
            }
            assert_every_persistent_state_has_a_complete_branch(problem, &branch.child);
        }
    }

    #[test]
    fn exact_persistent_policy_refines_one_fixed_state_and_identifies() {
        let problem = resolvable_fixture();
        let budget = QualificationBudgetV1::default();
        let receipt = solve_persistent_noise_policy(&problem, budget).unwrap();
        assert_eq!(receipt.root.worst_case_remaining, 1);
        assert_eq!(receipt.root.worst_case_cost, 2);
        assert_every_persistent_state_has_a_complete_branch(&problem, &receipt.root);
        let summary = verify_persistent_noise_policy(&problem, &receipt, budget).unwrap();
        assert!(summary.identified);
    }

    #[test]
    fn cross_state_signature_collisions_prevent_false_robust_identification() {
        let problem = cross_state_collision_fixture();
        let report = super::super::persistent_pair_identifiability_under_unknown_state(
            &problem, 0, 1,
        )
        .unwrap();
        assert!(report.indistinguishable_under_unknown_state);
        let receipt = solve_persistent_noise_policy(
            &problem,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(receipt.root.worst_case_remaining, 2);
        assert_eq!(receipt.root.worst_case_cost, 1);
        assert!(!receipt.root.is_identified());
        assert_every_persistent_state_has_a_complete_branch(&problem, &receipt.root);
    }

    #[test]
    fn no_probe_reuse_means_persistent_repeats_are_not_extra_evidence() {
        let problem = resolvable_fixture();
        let receipt = solve_persistent_noise_policy(&problem, QualificationBudgetV1::default()).unwrap();
        fn seen_probes(node: &PersistentPolicyNodeV1, path: &mut Vec<String>) {
            if let Some(probe_id) = &node.probe_id {
                assert!(!path.contains(probe_id));
                path.push(probe_id.clone());
                for branch in &node.branches {
                    seen_probes(&branch.child, path);
                }
                path.pop();
            }
        }
        seen_probes(&receipt.root, &mut Vec::new());
    }

    #[test]
    fn budgets_and_receipt_identity_fail_closed() {
        let problem = resolvable_fixture();
        let budget = QualificationBudgetV1::default();
        assert!(matches!(
            solve_persistent_noise_policy(
                &problem,
                QualificationBudgetV1 {
                    max_search_nodes: 1,
                    ..budget
                }
            ),
            Err(PersistentPlannerErrorV1::SearchNodeBudgetExceeded { .. })
        ));

        let shallow = solve_persistent_noise_policy(
            &problem,
            QualificationBudgetV1 { max_depth: 1, ..budget },
        )
        .unwrap();
        assert!(shallow.root.worst_case_remaining > 1);

        let no_cost = solve_persistent_noise_policy(
            &problem,
            QualificationBudgetV1 { max_total_cost: 0, ..budget },
        )
        .unwrap();
        assert_eq!(no_cost.root.worst_case_remaining, 2);
        assert_eq!(no_cost.root.leaf_reason, Some(PersistentLeafReasonV1::CostLimit));

        let receipt = solve_persistent_noise_policy(&problem, budget).unwrap();
        verify_persistent_noise_policy(&problem, &receipt, budget).unwrap();
        let changed_budget = QualificationBudgetV1 { max_total_cost: budget.max_total_cost - 1, ..budget };
        assert_eq!(
            verify_persistent_noise_policy(&problem, &receipt, changed_budget),
            Err(PersistentPlannerErrorV1::ReceiptMismatch)
        );
        let mut forged = receipt.clone();
        forged.root.worst_case_remaining = 2;
        assert_eq!(
            verify_persistent_noise_policy(&problem, &forged, budget),
            Err(PersistentPlannerErrorV1::ReceiptMismatch)
        );
        let mut changed = problem.clone();
        changed.predicted_outcomes_by_hypothesis[0][0][0] = 99;
        assert_eq!(
            verify_persistent_noise_policy(&changed, &receipt, budget),
            Err(PersistentPlannerErrorV1::ReceiptMismatch)
        );
    }
}
