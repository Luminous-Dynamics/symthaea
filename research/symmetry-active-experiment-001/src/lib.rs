// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact finite active-experiment planning for competing symmetry hypotheses.
//!
//! A probe may eliminate a hypothesis only when the frozen deterministic
//! observation table predicts a different outcome. The resulting plan is a
//! decision policy, not proof that any surviving hypothesis is a true
//! physical symmetry.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const MAX_HYPOTHESES: usize = 16;
const MAX_PROBES: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExperimentProbeV1 {
    pub probe_id: String,
    /// Positive integer acquisition/computation cost.
    pub cost: u64,
    /// Inadmissible probes are never selected.
    pub admissible: bool,
    /// One deterministic outcome label per hypothesis, by index.
    pub outcomes: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveExperimentProblemV1 {
    pub problem_id: String,
    /// Stable candidate order; probe outcome indexes refer to this order.
    pub hypotheses: Vec<String>,
    pub probes: Vec<ExperimentProbeV1>,
}

impl ActiveExperimentProblemV1 {
    /// Versioned canonical bytes. This format is deterministic but is not a
    /// cryptographic digest or signature.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"luminous.symm_active_experiment.v1\0".to_vec();
        append_string(&mut bytes, &self.problem_id);
        append_u64(&mut bytes, self.hypotheses.len() as u64);
        for hypothesis in &self.hypotheses {
            append_string(&mut bytes, hypothesis);
        }
        append_u64(&mut bytes, self.probes.len() as u64);
        for probe in &self.probes {
            append_string(&mut bytes, &probe.probe_id);
            append_u64(&mut bytes, probe.cost);
            bytes.push(u8::from(probe.admissible));
            append_u64(&mut bytes, probe.outcomes.len() as u64);
            for outcome in &probe.outcomes {
                bytes.extend_from_slice(&outcome.to_le_bytes());
            }
        }
        bytes
    }

    pub fn full_hypothesis_mask(&self) -> u64 {
        (1_u64 << self.hypotheses.len()) - 1
    }
}

fn append_u64(target: &mut Vec<u8>, value: u64) {
    target.extend_from_slice(&value.to_le_bytes());
}

fn append_string(target: &mut Vec<u8>, value: &str) {
    append_u64(target, value.len() as u64);
    target.extend_from_slice(value.as_bytes());
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualificationBudgetV1 {
    pub maximum_hypotheses: usize,
    pub maximum_probes: usize,
    pub maximum_work_units: u64,
}

impl Default for QualificationBudgetV1 {
    fn default() -> Self {
        Self {
            maximum_hypotheses: MAX_HYPOTHESES,
            maximum_probes: MAX_PROBES,
            maximum_work_units: 5_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanDispositionV1 {
    Identified {
        hypothesis_index: usize,
        hypothesis_id: String,
    },
    Probe {
        probe_index: usize,
        probe_id: String,
        probe_cost: u64,
        branches: Vec<OutcomeBranchV1>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutcomeBranchV1 {
    pub outcome: u32,
    pub child: Box<PlanNodeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanNodeV1 {
    /// Bit i is set iff hypothesis i is still compatible at this node.
    pub hypotheses_mask: u64,
    /// Minimum worst-case remaining cost from this node.
    pub worst_case_remaining_cost: u64,
    pub disposition: PlanDispositionV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveExperimentCertificateV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub problem_contract_bytes: Vec<u8>,
    pub hypothesis_count: usize,
    pub probe_count: usize,
    pub root_worst_case_cost: u64,
    pub explored_subsets: usize,
    pub work_units: u64,
    pub node_count: usize,
    pub root: PlanNodeV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperimentErrorV1 {
    InvalidProblemId,
    EmptyHypothesisSet,
    TooManyHypotheses { required: usize, maximum: usize },
    TooManyProbes { required: usize, maximum: usize },
    InvalidHypothesisId,
    DuplicateHypothesisId,
    InvalidProbeId,
    DuplicateProbeId,
    ZeroProbeCost,
    ProbeOutcomeCountMismatch { probe_index: usize },
    EmptyCandidateMask,
    WorkBudgetExceeded,
    CostOverflow,
    NoCompleteDiscriminationPolicy,
    ReceiptMismatch,
    InvalidObservationSequence,
    UnknownObservedOutcome { probe_id: String, outcome: u32 },
}

impl fmt::Display for ExperimentErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ExperimentErrorV1 {}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn validate_problem(
    problem: &ActiveExperimentProblemV1,
    budget: QualificationBudgetV1,
) -> Result<(), ExperimentErrorV1> {
    if !nonempty(&problem.problem_id) {
        return Err(ExperimentErrorV1::InvalidProblemId);
    }
    let hypothesis_count = problem.hypotheses.len();
    if hypothesis_count == 0 {
        return Err(ExperimentErrorV1::EmptyHypothesisSet);
    }
    if hypothesis_count > budget.maximum_hypotheses || hypothesis_count > MAX_HYPOTHESES {
        return Err(ExperimentErrorV1::TooManyHypotheses {
            required: hypothesis_count,
            maximum: budget.maximum_hypotheses.min(MAX_HYPOTHESES),
        });
    }
    if problem.probes.len() > budget.maximum_probes || problem.probes.len() > MAX_PROBES {
        return Err(ExperimentErrorV1::TooManyProbes {
            required: problem.probes.len(),
            maximum: budget.maximum_probes.min(MAX_PROBES),
        });
    }

    let mut hypothesis_ids = BTreeSet::new();
    for hypothesis in &problem.hypotheses {
        if !nonempty(hypothesis) {
            return Err(ExperimentErrorV1::InvalidHypothesisId);
        }
        if !hypothesis_ids.insert(hypothesis) {
            return Err(ExperimentErrorV1::DuplicateHypothesisId);
        }
    }

    let mut probe_ids = BTreeSet::new();
    for (probe_index, probe) in problem.probes.iter().enumerate() {
        if !nonempty(&probe.probe_id) {
            return Err(ExperimentErrorV1::InvalidProbeId);
        }
        if !probe_ids.insert(&probe.probe_id) {
            return Err(ExperimentErrorV1::DuplicateProbeId);
        }
        if probe.cost == 0 {
            return Err(ExperimentErrorV1::ZeroProbeCost);
        }
        if probe.outcomes.len() != hypothesis_count {
            return Err(ExperimentErrorV1::ProbeOutcomeCountMismatch { probe_index });
        }
    }
    Ok(())
}

fn partition_mask(
    problem: &ActiveExperimentProblemV1,
    probe_index: usize,
    hypotheses_mask: u64,
) -> BTreeMap<u32, u64> {
    let probe = &problem.probes[probe_index];
    let mut partitions = BTreeMap::<u32, u64>::new();
    let mut remaining_mask = hypotheses_mask;
    while remaining_mask != 0 {
        let hypothesis_index = remaining_mask.trailing_zeros() as usize;
        let bit = 1_u64 << hypothesis_index;
        remaining_mask &= remaining_mask - 1;
        let outcome = probe.outcomes[hypothesis_index];
        *partitions.entry(outcome).or_insert(0) |= bit;
    }
    partitions
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChoiceV1 {
    total_cost: u64,
    probe_index: usize,
}

fn solve_mask(
    problem: &ActiveExperimentProblemV1,
    hypotheses_mask: u64,
    budget: QualificationBudgetV1,
    memo: &mut BTreeMap<u64, Option<ChoiceV1>>,
    work_units: &mut u64,
) -> Result<Option<ChoiceV1>, ExperimentErrorV1> {
    let remaining = hypotheses_mask.count_ones();
    if remaining == 0 {
        return Err(ExperimentErrorV1::EmptyCandidateMask);
    }
    if remaining == 1 {
        return Ok(Some(ChoiceV1 {
            total_cost: 0,
            probe_index: usize::MAX,
        }));
    }
    if let Some(cached) = memo.get(&hypotheses_mask) {
        return Ok(*cached);
    }

    let mut best: Option<ChoiceV1> = None;
    for (probe_index, probe) in problem.probes.iter().enumerate() {
        if !probe.admissible {
            continue;
        }
        let partitions = partition_mask(problem, probe_index, hypotheses_mask);
        if partitions.len() < 2 {
            continue;
        }

        let partition_work = u64::from(remaining)
            .checked_mul(partitions.len() as u64)
            .ok_or(ExperimentErrorV1::WorkBudgetExceeded)?;
        *work_units = (*work_units)
            .checked_add(partition_work)
            .ok_or(ExperimentErrorV1::WorkBudgetExceeded)?;
        if *work_units > budget.maximum_work_units {
            return Err(ExperimentErrorV1::WorkBudgetExceeded);
        }

        let mut worst_child_cost = 0_u64;
        let mut all_branches_solvable = true;
        for child_mask in partitions.values() {
            let Some(child_choice) = solve_mask(problem, *child_mask, budget, memo, work_units)?
            else {
                all_branches_solvable = false;
                break;
            };
            worst_child_cost = worst_child_cost.max(child_choice.total_cost);
        }
        if !all_branches_solvable {
            continue;
        }

        let total_cost = probe
            .cost
            .checked_add(worst_child_cost)
            .ok_or(ExperimentErrorV1::CostOverflow)?;
        let candidate = ChoiceV1 {
            total_cost,
            probe_index,
        };
        let is_better = match best {
            None => true,
            Some(current) => {
                candidate.total_cost < current.total_cost
                    || (candidate.total_cost == current.total_cost
                        && candidate.probe_index < current.probe_index)
            }
        };
        if is_better {
            best = Some(candidate);
        }
    }

    memo.insert(hypotheses_mask, best);
    Ok(best)
}

fn build_node(
    problem: &ActiveExperimentProblemV1,
    hypotheses_mask: u64,
    memo: &BTreeMap<u64, Option<ChoiceV1>>,
) -> Result<PlanNodeV1, ExperimentErrorV1> {
    if hypotheses_mask.count_ones() == 1 {
        let hypothesis_index = hypotheses_mask.trailing_zeros() as usize;
        return Ok(PlanNodeV1 {
            hypotheses_mask,
            worst_case_remaining_cost: 0,
            disposition: PlanDispositionV1::Identified {
                hypothesis_index,
                hypothesis_id: problem.hypotheses[hypothesis_index].clone(),
            },
        });
    }

    let choice = memo
        .get(&hypotheses_mask)
        .copied()
        .flatten()
        .ok_or(ExperimentErrorV1::NoCompleteDiscriminationPolicy)?;
    let probe = &problem.probes[choice.probe_index];
    let partitions = partition_mask(problem, choice.probe_index, hypotheses_mask);
    let mut branches = Vec::with_capacity(partitions.len());
    let mut worst_child_cost = 0_u64;
    for (outcome, child_mask) in partitions {
        let child = build_node(problem, child_mask, memo)?;
        worst_child_cost = worst_child_cost.max(child.worst_case_remaining_cost);
        branches.push(OutcomeBranchV1 {
            outcome,
            child: Box::new(child),
        });
    }
    let total_cost = probe
        .cost
        .checked_add(worst_child_cost)
        .ok_or(ExperimentErrorV1::CostOverflow)?;
    if total_cost != choice.total_cost {
        return Err(ExperimentErrorV1::ReceiptMismatch);
    }
    Ok(PlanNodeV1 {
        hypotheses_mask,
        worst_case_remaining_cost: total_cost,
        disposition: PlanDispositionV1::Probe {
            probe_index: choice.probe_index,
            probe_id: probe.probe_id.clone(),
            probe_cost: probe.cost,
            branches,
        },
    })
}

fn count_nodes(node: &PlanNodeV1) -> usize {
    match &node.disposition {
        PlanDispositionV1::Identified { .. } => 1,
        PlanDispositionV1::Probe { branches, .. } => {
            1 + branches
                .iter()
                .map(|branch| count_nodes(&branch.child))
                .sum::<usize>()
        }
    }
}

pub fn compile_active_experiment(
    problem: &ActiveExperimentProblemV1,
    budget: QualificationBudgetV1,
) -> Result<ActiveExperimentCertificateV1, ExperimentErrorV1> {
    validate_problem(problem, budget)?;
    let root_mask = problem.full_hypothesis_mask();
    let mut memo = BTreeMap::new();
    let mut work_units = 0_u64;
    let choice = solve_mask(problem, root_mask, budget, &mut memo, &mut work_units)?
        .ok_or(ExperimentErrorV1::NoCompleteDiscriminationPolicy)?;
    let root = build_node(problem, root_mask, &memo)?;
    if root.worst_case_remaining_cost != choice.total_cost {
        return Err(ExperimentErrorV1::ReceiptMismatch);
    }
    Ok(ActiveExperimentCertificateV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        problem_contract_bytes: problem.canonical_bytes(),
        hypothesis_count: problem.hypotheses.len(),
        probe_count: problem.probes.len(),
        root_worst_case_cost: choice.total_cost,
        explored_subsets: memo.len(),
        work_units,
        node_count: count_nodes(&root),
        root,
    })
}

/// Independent replay recompiles the exact minimax decision tree from the
/// original problem and compares every tree node and accounting field.
pub fn verify_active_experiment_certificate(
    problem: &ActiveExperimentProblemV1,
    certificate: &ActiveExperimentCertificateV1,
    budget: QualificationBudgetV1,
) -> Result<(), ExperimentErrorV1> {
    let replayed = compile_active_experiment(problem, budget)?;
    if *certificate != replayed {
        return Err(ExperimentErrorV1::ReceiptMismatch);
    }
    Ok(())
}

/// Follow a sequence of observed deterministic outcomes. The surviving
/// hypothesis set is returned even if observations stop before a singleton.
pub fn replay_observations(
    problem: &ActiveExperimentProblemV1,
    certificate: &ActiveExperimentCertificateV1,
    observed_outcomes: &[u32],
    budget: QualificationBudgetV1,
) -> Result<Vec<String>, ExperimentErrorV1> {
    verify_active_experiment_certificate(problem, certificate, budget)?;
    let mut node = &certificate.root;
    for outcome in observed_outcomes {
        match &node.disposition {
            PlanDispositionV1::Identified { .. } => {
                return Err(ExperimentErrorV1::InvalidObservationSequence);
            }
            PlanDispositionV1::Probe {
                probe_id, branches, ..
            } => {
                let branch = branches
                    .iter()
                    .find(|branch| branch.outcome == *outcome)
                    .ok_or_else(|| ExperimentErrorV1::UnknownObservedOutcome {
                        probe_id: probe_id.clone(),
                        outcome: *outcome,
                    })?;
                node = &branch.child;
            }
        }
    }
    Ok((0..problem.hypotheses.len())
        .filter(|index| node.hypotheses_mask & (1_u64 << index) != 0)
        .map(|index| problem.hypotheses[index].clone())
        .collect())
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeRecommendationV1 {
    pub probe_index: usize,
    pub probe_id: String,
    pub worst_case_remaining_hypotheses: usize,
    pub outcome_entropy_bits: f64,
    pub cost: u64,
}

/// Non-authoritative greedy baseline: minimize the largest remaining outcome
/// class, then cost, then stable probe index.
pub fn choose_greedy_worst_case_probe(
    problem: &ActiveExperimentProblemV1,
    hypotheses_mask: u64,
) -> Option<ProbeRecommendationV1> {
    let mut best: Option<ProbeRecommendationV1> = None;
    for (probe_index, probe) in problem.probes.iter().enumerate() {
        if !probe.admissible {
            continue;
        }
        let partitions = partition_mask(problem, probe_index, hypotheses_mask);
        if partitions.len() < 2 {
            continue;
        }
        let largest = partitions
            .values()
            .map(|mask| mask.count_ones() as usize)
            .max()
            .unwrap_or(0);
        let recommendation = ProbeRecommendationV1 {
            probe_index,
            probe_id: probe.probe_id.clone(),
            worst_case_remaining_hypotheses: largest,
            outcome_entropy_bits: outcome_entropy(
                &partitions,
                hypotheses_mask.count_ones() as usize,
            ),
            cost: probe.cost,
        };
        let is_better = match &best {
            None => true,
            Some(current) => {
                recommendation.worst_case_remaining_hypotheses
                    < current.worst_case_remaining_hypotheses
                    || (recommendation.worst_case_remaining_hypotheses
                        == current.worst_case_remaining_hypotheses
                        && (recommendation.cost < current.cost
                            || (recommendation.cost == current.cost
                                && recommendation.probe_index < current.probe_index)))
            }
        };
        if is_better {
            best = Some(recommendation);
        }
    }
    best
}

/// Non-authoritative entropy baseline. The uniform prior is an explicit
/// assumption, not a posterior distribution or a pruning certificate.
pub fn choose_entropy_probe(
    problem: &ActiveExperimentProblemV1,
    hypotheses_mask: u64,
) -> Option<ProbeRecommendationV1> {
    let mut best: Option<ProbeRecommendationV1> = None;
    let hypothesis_count = hypotheses_mask.count_ones() as usize;
    if hypothesis_count == 0 {
        return None;
    }
    for (probe_index, probe) in problem.probes.iter().enumerate() {
        if !probe.admissible {
            continue;
        }
        let partitions = partition_mask(problem, probe_index, hypotheses_mask);
        if partitions.len() < 2 {
            continue;
        }
        let entropy = outcome_entropy(&partitions, hypothesis_count);
        let largest = partitions
            .values()
            .map(|mask| mask.count_ones() as usize)
            .max()
            .unwrap_or(0);
        let recommendation = ProbeRecommendationV1 {
            probe_index,
            probe_id: probe.probe_id.clone(),
            worst_case_remaining_hypotheses: largest,
            outcome_entropy_bits: entropy,
            cost: probe.cost,
        };
        let is_better = match &best {
            None => true,
            Some(current) => {
                recommendation.outcome_entropy_bits > current.outcome_entropy_bits
                    || (recommendation.outcome_entropy_bits == current.outcome_entropy_bits
                        && (recommendation.cost < current.cost
                            || (recommendation.cost == current.cost
                                && recommendation.probe_index < current.probe_index)))
            }
        };
        if is_better {
            best = Some(recommendation);
        }
    }
    best
}

fn outcome_entropy(partitions: &BTreeMap<u32, u64>, hypothesis_count: usize) -> f64 {
    if hypothesis_count == 0 {
        return 0.0;
    }
    let total = hypothesis_count as f64;
    partitions
        .values()
        .map(|mask| {
            let probability = mask.count_ones() as f64 / total;
            -probability * probability.log2()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> ActiveExperimentProblemV1 {
        ActiveExperimentProblemV1 {
            problem_id: "four-symmetry-hypotheses-v0.1".to_owned(),
            hypotheses: vec![
                "C2".to_owned(),
                "Cs".to_owned(),
                "D2".to_owned(),
                "C2v".to_owned(),
            ],
            probes: vec![
                ExperimentProbeV1 {
                    probe_id: "orientation-probe".to_owned(),
                    cost: 1,
                    admissible: true,
                    outcomes: vec![0, 0, 1, 1],
                },
                ExperimentProbeV1 {
                    probe_id: "boundary-probe".to_owned(),
                    cost: 1,
                    admissible: true,
                    outcomes: vec![0, 1, 1, 1],
                },
                ExperimentProbeV1 {
                    probe_id: "defect-localization-probe".to_owned(),
                    cost: 1,
                    admissible: true,
                    outcomes: vec![0, 0, 0, 1],
                },
            ],
        }
    }

    fn budget() -> QualificationBudgetV1 {
        QualificationBudgetV1::default()
    }

    fn cost_sensitive_fixture() -> ActiveExperimentProblemV1 {
        ActiveExperimentProblemV1 {
            problem_id: "cost-sensitive-four-hypotheses-v0.1".to_owned(),
            hypotheses: vec![
                "C2".to_owned(),
                "Cs".to_owned(),
                "D2".to_owned(),
                "C2v".to_owned(),
            ],
            probes: vec![
                ExperimentProbeV1 {
                    probe_id: "expensive-probe-a".to_owned(),
                    cost: 5,
                    admissible: true,
                    outcomes: vec![1, 2, 1, 0],
                },
                ExperimentProbeV1 {
                    probe_id: "expensive-probe-b".to_owned(),
                    cost: 5,
                    admissible: true,
                    outcomes: vec![0, 2, 1, 1],
                },
                ExperimentProbeV1 {
                    probe_id: "adaptive-optimal-probe".to_owned(),
                    cost: 5,
                    admissible: true,
                    outcomes: vec![2, 1, 0, 1],
                },
                ExperimentProbeV1 {
                    probe_id: "cheap-follow-up-probe".to_owned(),
                    cost: 2,
                    admissible: true,
                    outcomes: vec![0, 1, 0, 0],
                },
            ],
        }
    }

    fn greedy_policy_cost(problem: &ActiveExperimentProblemV1, mask: u64) -> u64 {
        if mask.count_ones() <= 1 {
            return 0;
        }
        let recommendation = choose_greedy_worst_case_probe(problem, mask)
            .expect("the frozen greedy fixture has a separating probe");
        let probe = &problem.probes[recommendation.probe_index];
        let mut partitions = BTreeMap::<u32, u64>::new();
        let mut remaining_mask = mask;
        while remaining_mask != 0 {
            let hypothesis_index = remaining_mask.trailing_zeros() as usize;
            let bit = 1_u64 << hypothesis_index;
            remaining_mask &= remaining_mask - 1;
            *partitions
                .entry(probe.outcomes[hypothesis_index])
                .or_insert(0) |= bit;
        }
        probe.cost
            + partitions
                .values()
                .map(|child_mask| greedy_policy_cost(problem, *child_mask))
                .max()
                .unwrap_or(0)
    }

    fn exhaustive_cost(problem: &ActiveExperimentProblemV1, mask: u64) -> Option<u64> {
        if mask.count_ones() <= 1 {
            return Some(0);
        }
        let mut best: Option<u64> = None;
        for probe in &problem.probes {
            if !probe.admissible {
                continue;
            }
            // Deliberately independent partition reconstruction for the
            // exhaustive small-instance oracle; do not call the production
            // partition_mask helper here.
            let mut partitions = BTreeMap::<u32, u64>::new();
            for (hypothesis_index, outcome) in probe.outcomes.iter().enumerate() {
                let bit = 1_u64 << hypothesis_index;
                if mask & bit != 0 {
                    *partitions.entry(*outcome).or_insert(0) |= bit;
                }
            }
            if partitions.len() < 2 {
                continue;
            }
            let mut worst = 0_u64;
            let mut possible = true;
            for child in partitions.values() {
                let Some(cost) = exhaustive_cost(problem, *child) else {
                    possible = false;
                    break;
                };
                worst = worst.max(cost);
            }
            if possible {
                let cost = probe.cost.checked_add(worst)?;
                best = Some(best.map_or(cost, |current| current.min(cost)));
            }
        }
        best
    }

    #[test]
    fn exact_minimax_matches_independent_exhaustive_oracle() {
        let problem = fixture();
        let certificate = compile_active_experiment(&problem, budget()).unwrap();
        assert_eq!(
            certificate.root_worst_case_cost,
            exhaustive_cost(&problem, problem.full_hypothesis_mask()).unwrap()
        );
        assert_eq!(certificate.root_worst_case_cost, 2);
        assert_eq!(certificate.node_count, 7);
        assert_eq!(certificate.explored_subsets, 6);
        verify_active_experiment_certificate(&problem, &certificate, budget()).unwrap();
    }

    #[test]
    fn exact_adaptive_minimax_beats_myopic_greedy_on_cost_sensitive_probes() {
        let problem = cost_sensitive_fixture();
        let certificate = compile_active_experiment(&problem, budget()).unwrap();
        let PlanDispositionV1::Probe {
            probe_index,
            probe_id,
            ..
        } = &certificate.root.disposition
        else {
            panic!("the root must select a probe");
        };
        assert_eq!(*probe_index, 2);
        assert_eq!(probe_id, "adaptive-optimal-probe");
        assert_eq!(certificate.root_worst_case_cost, 7);

        let greedy = greedy_policy_cost(&problem, problem.full_hypothesis_mask());
        assert_eq!(greedy, 10);
        assert!(certificate.root_worst_case_cost < greedy);

        let entropy = choose_entropy_probe(&problem, problem.full_hypothesis_mask()).unwrap();
        assert_eq!(entropy.probe_index, 0);
        assert_ne!(entropy.probe_index, *probe_index);
    }

    #[test]
    fn balanced_probe_is_selected_and_observations_identify_hypothesis() {
        let problem = fixture();
        let certificate = compile_active_experiment(&problem, budget()).unwrap();
        let PlanDispositionV1::Probe {
            probe_id, branches, ..
        } = &certificate.root.disposition
        else {
            panic!("root must select a probe");
        };
        assert_eq!(probe_id, "orientation-probe");
        assert_eq!(branches.len(), 2);
        assert_eq!(
            replay_observations(&problem, &certificate, &[0, 1], budget()).unwrap(),
            vec!["Cs".to_owned()]
        );
        assert_eq!(
            replay_observations(&problem, &certificate, &[1, 0], budget()).unwrap(),
            vec!["D2".to_owned()]
        );
    }

    #[test]
    fn entropy_and_greedy_are_explicit_non_authoritative_comparators() {
        let problem = fixture();
        let mask = problem.full_hypothesis_mask();
        let greedy = choose_greedy_worst_case_probe(&problem, mask).unwrap();
        let entropy = choose_entropy_probe(&problem, mask).unwrap();
        assert_eq!(greedy.probe_id, "orientation-probe");
        assert_eq!(entropy.probe_id, "orientation-probe");
        assert!((entropy.outcome_entropy_bits - 1.0).abs() <= 1.0e-12);
    }

    #[test]
    fn inadmissible_probe_cannot_be_selected_even_if_it_is_perfect() {
        let mut problem = fixture();
        problem.probes.push(ExperimentProbeV1 {
            probe_id: "forbidden-perfect-probe".to_owned(),
            cost: 1,
            admissible: false,
            outcomes: vec![0, 1, 2, 3],
        });
        let certificate = compile_active_experiment(&problem, budget()).unwrap();
        let PlanDispositionV1::Probe { probe_id, .. } = certificate.root.disposition else {
            panic!("root must select a probe");
        };
        assert_ne!(probe_id, "forbidden-perfect-probe");
    }

    #[test]
    fn no_discriminating_admissible_probe_fails_closed() {
        let mut problem = fixture();
        problem
            .probes
            .iter_mut()
            .for_each(|probe| probe.admissible = false);
        assert_eq!(
            compile_active_experiment(&problem, budget()),
            Err(ExperimentErrorV1::NoCompleteDiscriminationPolicy)
        );
    }

    #[test]
    fn forged_plan_or_mutated_observation_table_is_rejected() {
        let problem = fixture();
        let mut certificate = compile_active_experiment(&problem, budget()).unwrap();
        certificate.root_worst_case_cost += 1;
        assert_eq!(
            verify_active_experiment_certificate(&problem, &certificate, budget()),
            Err(ExperimentErrorV1::ReceiptMismatch)
        );

        let mut changed_problem = fixture();
        changed_problem.probes[0].outcomes[0] = 99;
        let original = compile_active_experiment(&fixture(), budget()).unwrap();
        assert_eq!(
            verify_active_experiment_certificate(&changed_problem, &original, budget()),
            Err(ExperimentErrorV1::ReceiptMismatch)
        );
    }

    #[test]
    fn unknown_outcomes_and_observations_after_identification_are_rejected() {
        let problem = fixture();
        let certificate = compile_active_experiment(&problem, budget()).unwrap();
        assert!(matches!(
            replay_observations(&problem, &certificate, &[99], budget()),
            Err(ExperimentErrorV1::UnknownObservedOutcome { .. })
        ));
        assert_eq!(
            replay_observations(&problem, &certificate, &[0, 1, 0], budget()),
            Err(ExperimentErrorV1::InvalidObservationSequence)
        );
    }

    #[test]
    fn malformed_inputs_and_work_exhaustion_fail_closed() {
        let mut duplicate = fixture();
        duplicate.hypotheses[1] = duplicate.hypotheses[0].clone();
        assert_eq!(
            compile_active_experiment(&duplicate, budget()),
            Err(ExperimentErrorV1::DuplicateHypothesisId)
        );

        let tiny = QualificationBudgetV1 {
            maximum_work_units: 1,
            ..budget()
        };
        assert_eq!(
            compile_active_experiment(&fixture(), tiny),
            Err(ExperimentErrorV1::WorkBudgetExceeded)
        );
    }

    #[test]
    fn canonical_contract_bytes_are_deterministic_and_bind_probe_costs() {
        let problem = fixture();
        let first = problem.canonical_bytes();
        let second = fixture().canonical_bytes();
        assert_eq!(first, second);

        let mut changed = fixture();
        changed.probes[0].cost += 1;
        assert_ne!(first, changed.canonical_bytes());
    }
}
