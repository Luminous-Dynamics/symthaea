// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact, deterministic active experiment planning for finite hypothesis sets.
//!
//! V0 is an experiment-design reference, not a physical experiment executor.
//! The prediction table is frozen before planning; observations may only
//! eliminate hypotheses whose declared deterministic prediction disagrees.

use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeV1 {
    pub probe_id: String,
    pub cost: u64,
    /// Predicted outcome for each hypothesis, aligned with hypothesis_ids.
    pub outcomes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteExperimentProblemV1 {
    pub problem_id: String,
    pub hypothesis_ids: Vec<String>,
    pub probes: Vec<ProbeV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualificationBudgetV1 {
    pub max_depth: u8,
    pub max_search_nodes: u64,
    pub max_total_cost: u64,
}

impl Default for QualificationBudgetV1 {
    fn default() -> Self {
        Self {
            max_depth: 8,
            max_search_nodes: 100_000,
            max_total_cost: u64::MAX,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProblemErrorV1 {
    EmptyProblemId,
    EmptyHypothesisSet,
    TooManyHypotheses,
    DuplicateHypothesisId,
    EmptyHypothesisId,
    EmptyProbeSet,
    TooManyProbes,
    EmptyProbeId,
    DuplicateProbeId,
    OutcomeCountMismatch { probe_id: String },
    EmptyOutcome { probe_id: String, hypothesis_index: usize },
}

impl fmt::Display for ProblemErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ProblemErrorV1 {}

impl FiniteExperimentProblemV1 {
    pub fn validate(&self) -> Result<(), ProblemErrorV1> {
        if self.problem_id.trim().is_empty() {
            return Err(ProblemErrorV1::EmptyProblemId);
        }
        if self.hypothesis_ids.is_empty() {
            return Err(ProblemErrorV1::EmptyHypothesisSet);
        }
        if self.hypothesis_ids.len() > 64 {
            return Err(ProblemErrorV1::TooManyHypotheses);
        }
        if self.probes.is_empty() {
            return Err(ProblemErrorV1::EmptyProbeSet);
        }
        if self.probes.len() > 64 {
            return Err(ProblemErrorV1::TooManyProbes);
        }

        let mut hypothesis_ids = self.hypothesis_ids.clone();
        hypothesis_ids.sort();
        if hypothesis_ids.iter().any(|id| id.trim().is_empty()) {
            return Err(ProblemErrorV1::EmptyHypothesisId);
        }
        if hypothesis_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ProblemErrorV1::DuplicateHypothesisId);
        }

        let mut probe_ids: Vec<&str> = Vec::with_capacity(self.probes.len());
        for probe in &self.probes {
            if probe.probe_id.trim().is_empty() {
                return Err(ProblemErrorV1::EmptyProbeId);
            }
            probe_ids.push(&probe.probe_id);
            if probe.outcomes.len() != self.hypothesis_ids.len() {
                return Err(ProblemErrorV1::OutcomeCountMismatch {
                    probe_id: probe.probe_id.clone(),
                });
            }
            if let Some(hypothesis_index) =
                probe.outcomes.iter().position(|outcome| outcome.trim().is_empty())
            {
                return Err(ProblemErrorV1::EmptyOutcome {
                    probe_id: probe.probe_id.clone(),
                    hypothesis_index,
                });
            }
        }
        probe_ids.sort_unstable();
        if probe_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ProblemErrorV1::DuplicateProbeId);
        }
        Ok(())
    }

    /// Canonical, versioned bytes for exact receipt replay.
    /// This encoding is not a cryptographic digest or signature.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"luminous.symm_active_experiment.v1\0".to_vec();
        append_bytes(&mut bytes, self.problem_id.as_bytes());
        bytes.extend_from_slice(&(self.hypothesis_ids.len() as u64).to_le_bytes());
        for hypothesis_id in &self.hypothesis_ids {
            append_bytes(&mut bytes, hypothesis_id.as_bytes());
        }
        bytes.extend_from_slice(&(self.probes.len() as u64).to_le_bytes());
        for probe in &self.probes {
            append_bytes(&mut bytes, probe.probe_id.as_bytes());
            bytes.extend_from_slice(&probe.cost.to_le_bytes());
            bytes.extend_from_slice(&(probe.outcomes.len() as u64).to_le_bytes());
            for outcome in &probe.outcomes {
                append_bytes(&mut bytes, outcome.as_bytes());
            }
        }
        bytes
    }
}

fn append_bytes(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(&(value.len() as u64).to_le_bytes());
    target.extend_from_slice(value);
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

fn partitions(
    problem: &FiniteExperimentProblemV1,
    active_mask: u64,
    probe_index: usize,
) -> BTreeMap<String, u64> {
    let mut groups = BTreeMap::new();
    for hypothesis_index in indices(active_mask, problem.hypothesis_ids.len()) {
        let outcome = &problem.probes[probe_index].outcomes[hypothesis_index];
        *groups.entry(outcome.clone()).or_insert(0) |= 1_u64 << hypothesis_index;
    }
    groups
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeafReasonV1 {
    Identified,
    DepthLimit,
    CostLimit,
    NoInformativeProbe,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyBranchV1 {
    pub outcome: String,
    pub child: Box<PolicyNodeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyNodeV1 {
    pub active_hypothesis_mask: u64,
    pub worst_case_remaining: usize,
    pub worst_case_cost: u64,
    pub probe_id: Option<String>,
    pub probe_cost: u64,
    pub leaf_reason: Option<LeafReasonV1>,
    pub branches: Vec<PolicyBranchV1>,
}

impl PolicyNodeV1 {
    fn leaf(active_mask: u64, reason: LeafReasonV1) -> Self {
        Self {
            active_hypothesis_mask: active_mask,
            worst_case_remaining: active_mask.count_ones() as usize,
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
pub struct GreedyChoiceV1 {
    pub probe_index: usize,
    pub worst_case_remaining: usize,
    pub partition_count: usize,
    pub cost: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlannerErrorV1 {
    InvalidProblem(ProblemErrorV1),
    InvalidBudget,
    InvalidHypothesisMask,
    InvalidProbeMask,
    SearchNodeBudgetExceeded { visited: u64, maximum: u64 },
    ArithmeticOverflow,
    InvalidHypothesisIndex,
    InvalidSchedule,
    MissingPredictedBranch { probe_id: String, outcome: String },
    TrueHypothesisEliminated,
    ReceiptMismatch,
}

impl fmt::Display for PlannerErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PlannerErrorV1 {}

fn validate_masks(
    problem: &FiniteExperimentProblemV1,
    active_mask: u64,
    available_probe_mask: u64,
) -> Result<(), PlannerErrorV1> {
    let valid_hypotheses = full_mask(problem.hypothesis_ids.len());
    let valid_probes = full_mask(problem.probes.len());
    if active_mask == 0 || active_mask & !valid_hypotheses != 0 {
        return Err(PlannerErrorV1::InvalidHypothesisMask);
    }
    if available_probe_mask & !valid_probes != 0 {
        return Err(PlannerErrorV1::InvalidProbeMask);
    }
    Ok(())
}

/// One-step minimax baseline: choose the affordable probe minimizing the
/// largest predicted outcome bucket. Ties prefer lower cost, then probe ID.
pub fn choose_greedy_minimax_probe(
    problem: &FiniteExperimentProblemV1,
    active_mask: u64,
    available_probe_mask: u64,
    remaining_cost: u64,
) -> Result<Option<GreedyChoiceV1>, PlannerErrorV1> {
    problem.validate().map_err(PlannerErrorV1::InvalidProblem)?;
    validate_masks(problem, active_mask, available_probe_mask)?;

    let mut best: Option<GreedyChoiceV1> = None;
    for probe_index in indices(available_probe_mask, problem.probes.len()) {
        let probe = &problem.probes[probe_index];
        if probe.cost > remaining_cost {
            continue;
        }
        let groups = partitions(problem, active_mask, probe_index);
        if groups.len() <= 1 {
            continue;
        }
        let worst_case_remaining = groups
            .values()
            .map(|mask| mask.count_ones() as usize)
            .max()
            .unwrap_or(0);
        let candidate = GreedyChoiceV1 {
            probe_index,
            worst_case_remaining,
            partition_count: groups.len(),
            cost: probe.cost,
        };
        let replace = best.is_none_or(|current| {
            (
                candidate.worst_case_remaining,
                candidate.cost,
                &probe.probe_id,
            ) < (
                current.worst_case_remaining,
                current.cost,
                &problem.probes[current.probe_index].probe_id,
            )
        });
        if replace {
            best = Some(candidate);
        }
    }
    Ok(best)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolverBudgetV1 {
    pub max_depth: u8,
    pub max_search_nodes: u64,
    pub max_total_cost: u64,
}

impl From<QualificationBudgetV1> for SolverBudgetV1 {
    fn from(value: QualificationBudgetV1) -> Self {
        Self {
            max_depth: value.max_depth,
            max_search_nodes: value.max_search_nodes,
            max_total_cost: value.max_total_cost,
        }
    }
}

fn solve_node(
    problem: &FiniteExperimentProblemV1,
    active_mask: u64,
    available_probe_mask: u64,
    depth_remaining: u8,
    cost_remaining: u64,
    budget: SolverBudgetV1,
    visited: &mut u64,
) -> Result<PolicyNodeV1, PlannerErrorV1> {
    *visited = visited
        .checked_add(1)
        .ok_or(PlannerErrorV1::ArithmeticOverflow)?;
    if *visited > budget.max_search_nodes {
        return Err(PlannerErrorV1::SearchNodeBudgetExceeded {
            visited: *visited,
            maximum: budget.max_search_nodes,
        });
    }

    if active_mask.count_ones() == 1 {
        return Ok(PolicyNodeV1::leaf(
            active_mask,
            LeafReasonV1::Identified,
        ));
    }
    if depth_remaining == 0 {
        return Ok(PolicyNodeV1::leaf(active_mask, LeafReasonV1::DepthLimit));
    }

    let mut best: Option<PolicyNodeV1> = None;
    let mut had_informative_probe = false;
    let mut had_affordable_probe = false;

    for probe_index in indices(available_probe_mask, problem.probes.len()) {
        let probe = &problem.probes[probe_index];
        let groups = partitions(problem, active_mask, probe_index);
        if groups.len() <= 1 {
            continue;
        }
        had_informative_probe = true;
        if probe.cost > cost_remaining {
            continue;
        }
        had_affordable_probe = true;

        let next_available = available_probe_mask & !(1_u64 << probe_index);
        let next_cost = cost_remaining - probe.cost;
        let mut branches = Vec::with_capacity(groups.len());
        let mut worst_case_remaining = 0usize;
        let mut worst_case_child_cost = 0u64;

        for (outcome, child_mask) in groups {
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
            branches.push(PolicyBranchV1 {
                outcome,
                child: Box::new(child),
            });
        }

        let worst_case_cost = probe
            .cost
            .checked_add(worst_case_child_cost)
            .ok_or(PlannerErrorV1::ArithmeticOverflow)?;
        let candidate = PolicyNodeV1 {
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
        LeafReasonV1::CostLimit
    } else {
        LeafReasonV1::NoInformativeProbe
    };
    Ok(PolicyNodeV1::leaf(active_mask, reason))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyReceiptV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub problem_contract_bytes: Vec<u8>,
    pub budget: SolverBudgetV1,
    pub root: PolicyNodeV1,
    pub search_nodes_visited: u64,
}

/// Exhaustive adaptive minimax search over all permitted finite probe trees.
///
/// Objective order:
/// 1. minimize worst-case surviving hypotheses;
/// 2. minimize worst-case cumulative probe cost;
/// 3. deterministic lexical tie-break on the first probe ID.
///
/// Exceeding the search-node budget returns an error; no partial plan is
/// returned or treated as qualified.
pub fn solve_exact_adaptive_policy(
    problem: &FiniteExperimentProblemV1,
    budget: QualificationBudgetV1,
) -> Result<PolicyReceiptV1, PlannerErrorV1> {
    problem.validate().map_err(PlannerErrorV1::InvalidProblem)?;
    if budget.max_depth == 0 || budget.max_search_nodes == 0 {
        return Err(PlannerErrorV1::InvalidBudget);
    }
    let solver_budget = SolverBudgetV1::from(budget);
    let active_mask = full_mask(problem.hypothesis_ids.len());
    let available_probe_mask = full_mask(problem.probes.len());
    let mut visited = 0u64;
    let root = solve_node(
        problem,
        active_mask,
        available_probe_mask,
        budget.max_depth,
        budget.max_total_cost,
        solver_budget,
        &mut visited,
    )?;

    Ok(PolicyReceiptV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        problem_contract_bytes: problem.canonical_bytes(),
        budget: solver_budget,
        root,
        search_nodes_visited: visited,
    })
}

/// Independent baseline for a fixed, non-adaptive probe schedule.
/// Returns the largest number of hypotheses sharing one complete outcome
/// signature. Schedule entries are probe indices and must be unique.
pub fn evaluate_nonadaptive_schedule(
    problem: &FiniteExperimentProblemV1,
    schedule: &[usize],
) -> Result<usize, PlannerErrorV1> {
    problem.validate().map_err(PlannerErrorV1::InvalidProblem)?;
    let mut seen = 0u64;
    for probe_index in schedule {
        if *probe_index >= problem.probes.len() || seen & (1_u64 << probe_index) != 0 {
            return Err(PlannerErrorV1::InvalidSchedule);
        }
        seen |= 1_u64 << probe_index;
    }

    let mut signatures: BTreeMap<Vec<String>, usize> = BTreeMap::new();
    for hypothesis_index in 0..problem.hypothesis_ids.len() {
        let signature: Vec<String> = schedule
            .iter()
            .map(|probe_index| problem.probes[*probe_index].outcomes[hypothesis_index].clone())
            .collect();
        *signatures.entry(signature).or_insert(0) += 1;
    }
    Ok(signatures.values().copied().max().unwrap_or(0))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HypothesisPathV1 {
    pub hypothesis_id: String,
    pub probe_ids: Vec<String>,
    pub terminal_hypothesis_ids: Vec<String>,
    pub eliminated_hypotheses: Vec<String>,
}

/// Follow one hypothesis through the policy using the frozen prediction table.
/// The true hypothesis must remain in every descendant mask and at the leaf.
pub fn replay_policy_for_hypothesis(
    problem: &FiniteExperimentProblemV1,
    receipt: &PolicyReceiptV1,
    hypothesis_index: usize,
) -> Result<HypothesisPathV1, PlannerErrorV1> {
    problem.validate().map_err(PlannerErrorV1::InvalidProblem)?;
    if hypothesis_index >= problem.hypothesis_ids.len() {
        return Err(PlannerErrorV1::InvalidHypothesisIndex);
    }
    if receipt.schema_version != 1
        || receipt.problem_id != problem.problem_id
        || receipt.problem_contract_bytes != problem.canonical_bytes()
    {
        return Err(PlannerErrorV1::ReceiptMismatch);
    }

    let true_bit = 1_u64 << hypothesis_index;
    let mut node = &receipt.root;
    let mut probe_ids = Vec::new();
    loop {
        if node.active_hypothesis_mask & true_bit == 0 {
            return Err(PlannerErrorV1::TrueHypothesisEliminated);
        }
        let Some(probe_id) = node.probe_id.as_ref() else {
            let terminal_hypothesis_ids = indices(
                node.active_hypothesis_mask,
                problem.hypothesis_ids.len(),
            )
            .map(|index| problem.hypothesis_ids[index].clone())
            .collect::<Vec<_>>();
            let eliminated_hypotheses = problem
                .hypothesis_ids
                .iter()
                .enumerate()
                .filter(|(index, _)| node.active_hypothesis_mask & (1_u64 << index) == 0)
                .map(|(_, id)| id.clone())
                .collect();
            return Ok(HypothesisPathV1 {
                hypothesis_id: problem.hypothesis_ids[hypothesis_index].clone(),
                probe_ids,
                terminal_hypothesis_ids,
                eliminated_hypotheses,
            });
        };

        let probe_index = problem
            .probes
            .iter()
            .position(|probe| probe.probe_id == *probe_id)
            .ok_or(PlannerErrorV1::ReceiptMismatch)?;
        let outcome = &problem.probes[probe_index].outcomes[hypothesis_index];
        let branch = node
            .branches
            .iter()
            .find(|branch| branch.outcome == *outcome)
            .ok_or_else(|| PlannerErrorV1::MissingPredictedBranch {
                probe_id: probe_id.clone(),
                outcome: outcome.clone(),
            })?;
        probe_ids.push(probe_id.clone());
        node = &branch.child;
    }
}

pub fn verify_policy_receipt(
    problem: &FiniteExperimentProblemV1,
    receipt: &PolicyReceiptV1,
) -> Result<(), PlannerErrorV1> {
    if receipt.schema_version != 1
        || receipt.problem_id != problem.problem_id
        || receipt.problem_contract_bytes != problem.canonical_bytes()
    {
        return Err(PlannerErrorV1::ReceiptMismatch);
    }
    let replayed = solve_exact_adaptive_policy(
        problem,
        QualificationBudgetV1 {
            max_depth: receipt.budget.max_depth,
            max_search_nodes: receipt.budget.max_search_nodes,
            max_total_cost: receipt.budget.max_total_cost,
        },
    )?;
    if *receipt != replayed {
        return Err(PlannerErrorV1::ReceiptMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> FiniteExperimentProblemV1 {
        FiniteExperimentProblemV1 {
            problem_id: "adaptive-d4-symmetry-falsification-v1".to_owned(),
            hypothesis_ids: vec![
                "D4-hypothesis-A".to_owned(),
                "D4-hypothesis-B".to_owned(),
                "D4-hypothesis-C".to_owned(),
                "D4-hypothesis-D".to_owned(),
            ],
            probes: vec![
                ProbeV1 {
                    probe_id: "p-balanced-first".to_owned(),
                    cost: 1,
                    outcomes: vec!["left".to_owned(), "left".to_owned(), "right".to_owned(), "right".to_owned()],
                },
                ProbeV1 {
                    probe_id: "p-separate-ab".to_owned(),
                    cost: 1,
                    outcomes: vec!["a".to_owned(), "b".to_owned(), "a".to_owned(), "a".to_owned()],
                },
                ProbeV1 {
                    probe_id: "p-separate-cd".to_owned(),
                    cost: 1,
                    outcomes: vec!["a".to_owned(), "a".to_owned(), "b".to_owned(), "a".to_owned()],
                },
            ],
        }
    }

    #[test]
    fn adaptive_two_probe_tree_beats_every_two_probe_nonadaptive_schedule() {
        let problem = fixture();
        let receipt = solve_exact_adaptive_policy(&problem, QualificationBudgetV1::default()).unwrap();

        assert_eq!(receipt.root.worst_case_remaining, 1);
        assert_eq!(receipt.root.worst_case_cost, 2);
        assert_eq!(receipt.root.probe_id.as_deref(), Some("p-balanced-first"));

        for first in 0..problem.probes.len() {
            for second in (first + 1)..problem.probes.len() {
                assert_eq!(
                    evaluate_nonadaptive_schedule(&problem, &[first, second]).unwrap(),
                    2
                );
            }
        }

        for hypothesis_index in 0..problem.hypothesis_ids.len() {
            let path = replay_policy_for_hypothesis(&problem, &receipt, hypothesis_index).unwrap();
            assert_eq!(path.terminal_hypothesis_ids.len(), 1);
            assert_eq!(path.terminal_hypothesis_ids[0], path.hypothesis_id);
            assert_eq!(path.probe_ids.len(), 2);
        }
    }

    #[test]
    fn greedy_one_step_selector_agrees_on_first_probe_but_has_no_adaptive_certificate() {
        let problem = fixture();
        let choice = choose_greedy_minimax_probe(
            &problem,
            full_mask(problem.hypothesis_ids.len()),
            full_mask(problem.probes.len()),
            u64::MAX,
        )
        .unwrap()
        .unwrap();
        assert_eq!(problem.probes[choice.probe_index].probe_id, "p-balanced-first");
        assert_eq!(choice.worst_case_remaining, 2);
        assert_eq!(choice.partition_count, 2);
    }

    #[test]
    fn identical_prediction_probe_is_never_selected() {
        let mut problem = fixture();
        problem.probes.push(ProbeV1 {
            probe_id: "p-no-information".to_owned(),
            cost: 0,
            outcomes: vec!["same".to_owned(); 4],
        });
        let choice = choose_greedy_minimax_probe(
            &problem,
            full_mask(4),
            full_mask(4),
            u64::MAX,
        )
        .unwrap()
        .unwrap();
        assert_ne!(problem.probes[choice.probe_index].probe_id, "p-no-information");
    }

    #[test]
    fn cost_limit_returns_an_unresolved_leaf_instead_of_inventing_progress() {
        let problem = fixture();
        let receipt = solve_exact_adaptive_policy(
            &problem,
            QualificationBudgetV1 {
                max_depth: 4,
                max_search_nodes: 100,
                max_total_cost: 0,
            },
        )
        .unwrap();
        assert_eq!(receipt.root.worst_case_remaining, 4);
        assert_eq!(receipt.root.leaf_reason, Some(LeafReasonV1::CostLimit));
    }

    #[test]
    fn search_node_exhaustion_fails_closed_without_a_partial_policy() {
        let problem = fixture();
        assert!(matches!(
            solve_exact_adaptive_policy(
                &problem,
                QualificationBudgetV1 {
                    max_depth: 8,
                    max_search_nodes: 1,
                    max_total_cost: u64::MAX,
                }
            ),
            Err(PlannerErrorV1::SearchNodeBudgetExceeded { .. })
        ));
    }

    #[test]
    fn policy_receipt_replays_exactly_and_forgery_is_rejected() {
        let problem = fixture();
        let mut receipt = solve_exact_adaptive_policy(&problem, QualificationBudgetV1::default()).unwrap();
        verify_policy_receipt(&problem, &receipt).unwrap();
        receipt.root.worst_case_remaining = 4;
        assert_eq!(
            verify_policy_receipt(&problem, &receipt),
            Err(PlannerErrorV1::ReceiptMismatch)
        );
    }

    #[test]
    fn mutated_predictions_invalidate_receipt_replay() {
        let problem = fixture();
        let receipt = solve_exact_adaptive_policy(&problem, QualificationBudgetV1::default()).unwrap();
        let mut mutated = problem.clone();
        mutated.probes[0].outcomes[0] = "forged".to_owned();
        assert_eq!(
            verify_policy_receipt(&mutated, &receipt),
            Err(PlannerErrorV1::ReceiptMismatch)
        );
    }

    #[test]
    fn malformed_and_ambiguous_problem_contracts_are_rejected() {
        let mut problem = fixture();
        problem.hypothesis_ids[1] = problem.hypothesis_ids[0].clone();
        assert_eq!(problem.validate(), Err(ProblemErrorV1::DuplicateHypothesisId));

        let mut problem = fixture();
        problem.probes[0].outcomes.pop();
        assert!(matches!(
            problem.validate(),
            Err(ProblemErrorV1::OutcomeCountMismatch { .. })
        ));
    }

    #[test]
    fn depth_limit_preserves_hypotheses_and_reports_unresolved_set() {
        let problem = fixture();
        let receipt = solve_exact_adaptive_policy(
            &problem,
            QualificationBudgetV1 {
                max_depth: 1,
                max_search_nodes: 100,
                max_total_cost: u64::MAX,
            },
        )
        .unwrap();
        assert_eq!(receipt.root.worst_case_remaining, 2);
        for hypothesis_index in 0..problem.hypothesis_ids.len() {
            let path = replay_policy_for_hypothesis(&problem, &receipt, hypothesis_index).unwrap();
            assert!(path.terminal_hypothesis_ids.contains(&path.hypothesis_id));
        }
    }
}
