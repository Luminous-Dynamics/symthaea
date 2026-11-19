// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact active identification of task-preserving subgroups of a known group.
//!
//! This crate enumerates valid subgroups from a validated finite multiplication
//! table, adapts element membership to the existing exact active-experiment
//! planner, and replays evidence-bound observations. It does not authorize
//! quotient pruning or prove that the supplied task checker is physically true.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use symmetry_active_experiment_reference::{
    compile_active_experiment, solve_exact_static_probe_design,
    ActiveExperimentCertificateV1, ActiveExperimentProblemV1, ExactStaticProbeCertificateV1,
    ExperimentErrorV1, ExperimentProbeV1, PlanDispositionV1, QualificationBudgetV1,
    StaticProbeBudgetV1,
};

const MAX_GROUP_ORDER: usize = 16;
const MAX_SUBGROUPS: usize = 16;
const MAX_PROBES: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteGroupTableV1 {
    pub group_id: String,
    pub element_ids: Vec<String>,
    pub identity_index: usize,
    /// Row-major multiplication: multiplication[left][right] = left * right.
    pub multiplication: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MembershipProbeV1 {
    pub probe_id: String,
    pub group_element_index: usize,
    pub task_contract_id: String,
    pub checker_id: String,
    pub cost: u64,
    pub admissible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupIdentificationProblemV1 {
    pub problem_id: String,
    pub group: FiniteGroupTableV1,
    pub probes: Vec<MembershipProbeV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubgroupBudgetV1 {
    pub maximum_group_order: usize,
    pub maximum_subgroups: usize,
    pub maximum_probes: usize,
    pub maximum_work_units: u64,
    pub maximum_static_subsets: u64,
}

impl Default for SubgroupBudgetV1 {
    fn default() -> Self {
        Self {
            maximum_group_order: MAX_GROUP_ORDER,
            maximum_subgroups: MAX_SUBGROUPS,
            maximum_probes: MAX_PROBES,
            maximum_work_units: 20_000_000,
            maximum_static_subsets: 1_048_576,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupHypothesisV1 {
    pub subgroup_id: String,
    pub element_indices: Vec<usize>,
    pub element_ids: Vec<String>,
    pub element_mask: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupEnumerationEvidenceV1 {
    pub group_axiom_checks: u64,
    pub raw_subset_masks_scanned: u64,
    pub candidate_subsets_examined: u64,
    pub identity_membership_checks: u64,
    pub inverse_closure_checks: u64,
    pub product_closure_checks: u64,
    pub total_work_units: u64,
    pub complete_subset_lattice_search: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupEnumerationV1 {
    pub schema_version: u32,
    pub group_id: String,
    pub group_contract_bytes: Vec<u8>,
    pub group_order: usize,
    pub subgroups: Vec<SubgroupHypothesisV1>,
    pub evidence: SubgroupEnumerationEvidenceV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubgroupErrorV1 {
    InvalidProblemId,
    InvalidGroupId,
    EmptyGroup,
    GroupOrderExceeded { required: usize, maximum: usize },
    MultiplicationTableNotSquare,
    InvalidElementId,
    DuplicateElementId,
    InvalidIdentity,
    ProductOutOfRange,
    NonAssociativeGroup,
    IdentityLawViolation,
    MissingInverse,
    WorkBudgetExceeded { required: u64, maximum: u64 },
    TooManySubgroups { required_at_least: usize, maximum: usize },
    EmptyProbeSet,
    TooManyProbes { required: usize, maximum: usize },
    InvalidProbeId,
    DuplicateProbeId,
    ProbeElementOutOfRange { probe_id: String },
    InvalidTaskContractId { probe_id: String },
    InvalidCheckerId { probe_id: String },
    ZeroProbeCost { probe_id: String },
    UnknownProbe { probe_id: String },
    InadmissibleProbe { probe_id: String },
    EvidenceContextMismatch { probe_id: String },
    MissingEvidenceReference { probe_id: String },
    InvalidObservationId,
    DuplicateObservationId,
    NoCandidateSubgroups,
    InvalidCandidateFrontier,
    ReceiptMismatch,
    Planner(ExperimentErrorV1),
}

impl fmt::Display for SubgroupErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SubgroupErrorV1 {}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn append_u64(target: &mut Vec<u8>, value: u64) {
    target.extend_from_slice(&value.to_le_bytes());
}

fn append_bytes(target: &mut Vec<u8>, value: &[u8]) {
    append_u64(target, value.len() as u64);
    target.extend_from_slice(value);
}

fn append_string(target: &mut Vec<u8>, value: &str) {
    append_bytes(target, value.as_bytes());
}

impl FiniteGroupTableV1 {
    /// Canonical v1 bytes; deterministic encoding, not a cryptographic digest.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"luminous.symm_subgroup_active.group.v1\0".to_vec();
        append_string(&mut bytes, &self.group_id);
        append_u64(&mut bytes, self.identity_index as u64);
        append_u64(&mut bytes, self.element_ids.len() as u64);
        for element_id in &self.element_ids {
            append_string(&mut bytes, element_id);
        }
        for row in &self.multiplication {
            append_u64(&mut bytes, row.len() as u64);
            for product in row {
                append_u64(&mut bytes, *product as u64);
            }
        }
        bytes
    }
}

impl SubgroupIdentificationProblemV1 {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"luminous.symm_subgroup_active.problem.v1\0".to_vec();
        append_string(&mut bytes, &self.problem_id);
        append_bytes(&mut bytes, &self.group.canonical_bytes());
        append_u64(&mut bytes, self.probes.len() as u64);
        for probe in &self.probes {
            append_string(&mut bytes, &probe.probe_id);
            append_u64(&mut bytes, probe.group_element_index as u64);
            append_string(&mut bytes, &probe.task_contract_id);
            append_string(&mut bytes, &probe.checker_id);
            append_u64(&mut bytes, probe.cost);
            bytes.push(u8::from(probe.admissible));
        }
        bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ValidatedGroupV1 {
    inverses: Vec<usize>,
    axiom_checks: u64,
}

fn validate_group(
    group: &FiniteGroupTableV1,
    budget: SubgroupBudgetV1,
) -> Result<ValidatedGroupV1, SubgroupErrorV1> {
    if !nonempty(&group.group_id) {
        return Err(SubgroupErrorV1::InvalidGroupId);
    }
    let order = group.element_ids.len();
    if order == 0 {
        return Err(SubgroupErrorV1::EmptyGroup);
    }
    if order > budget.maximum_group_order.min(MAX_GROUP_ORDER) {
        return Err(SubgroupErrorV1::GroupOrderExceeded {
            required: order,
            maximum: budget.maximum_group_order.min(MAX_GROUP_ORDER),
        });
    }
    if group.multiplication.len() != order
        || group.multiplication.iter().any(|row| row.len() != order)
    {
        return Err(SubgroupErrorV1::MultiplicationTableNotSquare);
    }
    if group.identity_index >= order {
        return Err(SubgroupErrorV1::InvalidIdentity);
    }
    let mut ids = BTreeSet::new();
    for id in &group.element_ids {
        if !nonempty(id) {
            return Err(SubgroupErrorV1::InvalidElementId);
        }
        if !ids.insert(id) {
            return Err(SubgroupErrorV1::DuplicateElementId);
        }
    }

    let mut checks = 0_u64;
    for row in 0..order {
        for column in 0..order {
            checks = checks.checked_add(1).ok_or(SubgroupErrorV1::WorkBudgetExceeded {
                required: u64::MAX,
                maximum: budget.maximum_work_units,
            })?;
            if group.multiplication[row][column] >= order {
                return Err(SubgroupErrorV1::ProductOutOfRange);
            }
        }
    }
    for left in 0..order {
        for middle in 0..order {
            for right in 0..order {
                checks = checks.checked_add(1).ok_or(SubgroupErrorV1::WorkBudgetExceeded {
                    required: u64::MAX,
                    maximum: budget.maximum_work_units,
                })?;
                let lhs = group.multiplication[group.multiplication[left][middle]][right];
                let rhs = group.multiplication[left][group.multiplication[middle][right]];
                if lhs != rhs {
                    return Err(SubgroupErrorV1::NonAssociativeGroup);
                }
            }
        }
    }

    for element in 0..order {
        checks = checks.checked_add(2).ok_or(SubgroupErrorV1::WorkBudgetExceeded {
            required: u64::MAX,
            maximum: budget.maximum_work_units,
        })?;
        if group.multiplication[group.identity_index][element] != element
            || group.multiplication[element][group.identity_index] != element
        {
            return Err(SubgroupErrorV1::IdentityLawViolation);
        }
    }

    let mut inverses = vec![usize::MAX; order];
    for element in 0..order {
        let inverse = (0..order).find(|candidate| {
            group.multiplication[element][*candidate] == group.identity_index
                && group.multiplication[*candidate][element] == group.identity_index
        });
        checks = checks.checked_add(order as u64).ok_or(SubgroupErrorV1::WorkBudgetExceeded {
            required: u64::MAX,
            maximum: budget.maximum_work_units,
        })?;
        inverses[element] = inverse.ok_or(SubgroupErrorV1::MissingInverse)?;
    }

    if checks > budget.maximum_work_units {
        return Err(SubgroupErrorV1::WorkBudgetExceeded {
            required: checks,
            maximum: budget.maximum_work_units,
        });
    }
    Ok(ValidatedGroupV1 {
        inverses,
        axiom_checks: checks,
    })
}

fn subgroup_id(_group: &FiniteGroupTableV1, element_indices: &[usize]) -> String {
    // The ordered index list is unambiguous even if caller-supplied element
    // labels contain commas, brackets, or other delimiter characters.
    let members = element_indices
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("H[{members}]")
}

/// Enumerate every subgroup by exhaustive subset closure under inverse and
/// multiplication. The subset search is complete for the declared finite
/// group; arbitrary subsets are never admitted as hypotheses.
pub fn enumerate_subgroups(
    group: &FiniteGroupTableV1,
    budget: SubgroupBudgetV1,
) -> Result<SubgroupEnumerationV1, SubgroupErrorV1> {
    let validated = validate_group(group, budget)?;
    let order = group.element_ids.len();
    let candidate_subset_count = 1_u64
        .checked_shl((order - 1) as u32)
        .ok_or(SubgroupErrorV1::WorkBudgetExceeded {
            required: u64::MAX,
            maximum: budget.maximum_work_units,
        })?;
    let raw_subset_masks_scanned = 1_u64.checked_shl(order as u32).ok_or(
        SubgroupErrorV1::WorkBudgetExceeded {
            required: u64::MAX,
            maximum: budget.maximum_work_units,
        },
    )?;
    let order_squared = (order as u64)
        .checked_mul(order as u64)
        .ok_or(SubgroupErrorV1::WorkBudgetExceeded {
            required: u64::MAX,
            maximum: budget.maximum_work_units,
        })?;
    let worst_case_subset_work = candidate_subset_count
        .checked_mul(order_squared + order as u64)
        .and_then(|value| value.checked_add(raw_subset_masks_scanned))
        .and_then(|value| value.checked_add(validated.axiom_checks))
        .ok_or(SubgroupErrorV1::WorkBudgetExceeded {
            required: u64::MAX,
            maximum: budget.maximum_work_units,
        })?;
    if worst_case_subset_work > budget.maximum_work_units {
        return Err(SubgroupErrorV1::WorkBudgetExceeded {
            required: worst_case_subset_work,
            maximum: budget.maximum_work_units,
        });
    }

    let mut subgroups = Vec::new();
    let mut subsets_examined = 0_u64;
    let mut inverse_checks = 0_u64;
    let mut product_checks = 0_u64;
    let identity_bit = 1_u64 << group.identity_index;
    let all_masks = 1_u64 << order;

    for mask in 0..all_masks {
        if mask & identity_bit == 0 {
            continue;
        }
        subsets_examined += 1;
        let mut is_subgroup = true;

        for left in 0..order {
            if mask & (1_u64 << left) == 0 {
                continue;
            }
            inverse_checks += 1;
            if mask & (1_u64 << validated.inverses[left]) == 0 {
                is_subgroup = false;
                break;
            }
            for right in 0..order {
                if mask & (1_u64 << right) == 0 {
                    continue;
                }
                product_checks += 1;
                let product = group.multiplication[left][right];
                if mask & (1_u64 << product) == 0 {
                    is_subgroup = false;
                    break;
                }
            }
            if !is_subgroup {
                break;
            }
        }

        if is_subgroup {
            if subgroups.len() >= budget.maximum_subgroups.min(MAX_SUBGROUPS) {
                return Err(SubgroupErrorV1::TooManySubgroups {
                    required_at_least: subgroups.len() + 1,
                    maximum: budget.maximum_subgroups.min(MAX_SUBGROUPS),
                });
            }
            let element_indices = (0..order)
                .filter(|index| mask & (1_u64 << index) != 0)
                .collect::<Vec<_>>();
            let element_ids = element_indices
                .iter()
                .map(|index| group.element_ids[*index].clone())
                .collect::<Vec<_>>();
            subgroups.push(SubgroupHypothesisV1 {
                subgroup_id: subgroup_id(group, &element_indices),
                element_indices,
                element_ids,
                element_mask: mask,
            });
        }
    }

    let total_work_units = validated
        .axiom_checks
        .checked_add(raw_subset_masks_scanned)
        .and_then(|value| value.checked_add(inverse_checks))
        .and_then(|value| value.checked_add(product_checks))
        .ok_or(SubgroupErrorV1::WorkBudgetExceeded {
            required: u64::MAX,
            maximum: budget.maximum_work_units,
        })?;
    if total_work_units > budget.maximum_work_units {
        return Err(SubgroupErrorV1::WorkBudgetExceeded {
            required: total_work_units,
            maximum: budget.maximum_work_units,
        });
    }

    Ok(SubgroupEnumerationV1 {
        schema_version: 1,
        group_id: group.group_id.clone(),
        group_contract_bytes: group.canonical_bytes(),
        group_order: order,
        subgroups,
        evidence: SubgroupEnumerationEvidenceV1 {
            group_axiom_checks: validated.axiom_checks,
            raw_subset_masks_scanned,
            candidate_subsets_examined: subsets_examined,
            identity_membership_checks: raw_subset_masks_scanned,
            inverse_closure_checks: inverse_checks,
            product_closure_checks: product_checks,
            total_work_units,
            complete_subset_lattice_search: true,
        },
    })
}

fn validate_problem(
    problem: &SubgroupIdentificationProblemV1,
    budget: SubgroupBudgetV1,
) -> Result<(), SubgroupErrorV1> {
    if !nonempty(&problem.problem_id) {
        return Err(SubgroupErrorV1::InvalidProblemId);
    }
    validate_group(&problem.group, budget)?;
    if problem.probes.is_empty() {
        return Err(SubgroupErrorV1::EmptyProbeSet);
    }
    if problem.probes.len() > budget.maximum_probes.min(MAX_PROBES) {
        return Err(SubgroupErrorV1::TooManyProbes {
            required: problem.probes.len(),
            maximum: budget.maximum_probes.min(MAX_PROBES),
        });
    }
    let mut probe_ids = BTreeSet::new();
    for probe in &problem.probes {
        if !nonempty(&probe.probe_id) {
            return Err(SubgroupErrorV1::InvalidProbeId);
        }
        if !probe_ids.insert(&probe.probe_id) {
            return Err(SubgroupErrorV1::DuplicateProbeId);
        }
        if probe.group_element_index >= problem.group.element_ids.len() {
            return Err(SubgroupErrorV1::ProbeElementOutOfRange {
                probe_id: probe.probe_id.clone(),
            });
        }
        if !nonempty(&probe.task_contract_id) {
            return Err(SubgroupErrorV1::InvalidTaskContractId {
                probe_id: probe.probe_id.clone(),
            });
        }
        if !nonempty(&probe.checker_id) {
            return Err(SubgroupErrorV1::InvalidCheckerId {
                probe_id: probe.probe_id.clone(),
            });
        }
        if probe.cost == 0 {
            return Err(SubgroupErrorV1::ZeroProbeCost {
                probe_id: probe.probe_id.clone(),
            });
        }
    }
    Ok(())
}

fn build_active_problem(
    problem: &SubgroupIdentificationProblemV1,
    enumeration: &SubgroupEnumerationV1,
) -> ActiveExperimentProblemV1 {
    ActiveExperimentProblemV1 {
        problem_id: format!("{}::subgroup-membership", problem.problem_id),
        hypotheses: enumeration
            .subgroups
            .iter()
            .map(|subgroup| subgroup.subgroup_id.clone())
            .collect(),
        probes: problem
            .probes
            .iter()
            .map(|probe| ExperimentProbeV1 {
                probe_id: probe.probe_id.clone(),
                cost: probe.cost,
                admissible: probe.admissible,
                outcomes: enumeration
                    .subgroups
                    .iter()
                    .map(|subgroup| {
                        u32::from(
                            subgroup.element_mask & (1_u64 << probe.group_element_index) != 0,
                        )
                    })
                    .collect(),
            })
            .collect(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdaptivePolicyV1 {
    Complete(ActiveExperimentCertificateV1),
    Incomplete {
        unresolved_pairs: Vec<(String, String)>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupCostComparisonV1 {
    pub full_scan_cost: u64,
    pub static_basis_cost: u64,
    pub adaptive_worst_case_cost: Option<u64>,
    /// Sum of per-hypothesis path costs under a uniform subgroup prior.
    pub uniform_prior_cost_sum: u64,
    pub uniform_prior_hypothesis_count: usize,
}

fn policy_path_cost(
    problem: &ActiveExperimentProblemV1,
    policy: &ActiveExperimentCertificateV1,
    hypothesis_index: usize,
) -> Result<u64, SubgroupErrorV1> {
    let mut node = &policy.root;
    let mut cost = 0_u64;
    loop {
        match &node.disposition {
            PlanDispositionV1::Identified { hypothesis_index: identified, .. } => {
                if *identified != hypothesis_index {
                    return Err(SubgroupErrorV1::ReceiptMismatch);
                }
                return Ok(cost);
            }
            PlanDispositionV1::Probe { probe_index, probe_cost, branches, .. } => {
                let outcome = problem
                    .probes
                    .get(*probe_index)
                    .and_then(|probe| probe.outcomes.get(hypothesis_index))
                    .ok_or(SubgroupErrorV1::ReceiptMismatch)?;
                let branch = branches
                    .iter()
                    .find(|branch| branch.outcome == *outcome)
                    .ok_or(SubgroupErrorV1::ReceiptMismatch)?;
                cost = cost.checked_add(*probe_cost).ok_or(
                    SubgroupErrorV1::Planner(ExperimentErrorV1::CostOverflow),
                )?;
                node = &branch.child;
            }
        }
    }
}

fn build_cost_comparison(
    problem: &SubgroupIdentificationProblemV1,
    active_problem: &ActiveExperimentProblemV1,
    adaptive_policy: &AdaptivePolicyV1,
    static_probe_certificate: &ExactStaticProbeCertificateV1,
) -> Result<SubgroupCostComparisonV1, SubgroupErrorV1> {
    let full_scan_cost = problem
        .probes
        .iter()
        .filter(|probe| probe.admissible)
        .try_fold(0_u64, |total, probe| total.checked_add(probe.cost))
        .ok_or(SubgroupErrorV1::Planner(ExperimentErrorV1::CostOverflow))?;
    let mut uniform_prior_cost_sum = 0_u64;
    let adaptive_worst_case_cost = match adaptive_policy {
        AdaptivePolicyV1::Complete(policy) => {
            for hypothesis_index in 0..active_problem.hypotheses.len() {
                uniform_prior_cost_sum = uniform_prior_cost_sum
                    .checked_add(policy_path_cost(active_problem, policy, hypothesis_index)?)
                    .ok_or(SubgroupErrorV1::Planner(ExperimentErrorV1::CostOverflow))?;
            }
            Some(policy.root_worst_case_cost)
        }
        AdaptivePolicyV1::Incomplete { .. } => None,
    };
    Ok(SubgroupCostComparisonV1 {
        full_scan_cost,
        static_basis_cost: static_probe_certificate.selected_cost,
        adaptive_worst_case_cost,
        uniform_prior_cost_sum,
        uniform_prior_hypothesis_count: active_problem.hypotheses.len(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupIdentificationCertificateV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub problem_contract_bytes: Vec<u8>,
    pub enumeration: SubgroupEnumerationV1,
    pub active_problem: ActiveExperimentProblemV1,
    pub adaptive_policy: AdaptivePolicyV1,
    pub static_probe_certificate: ExactStaticProbeCertificateV1,
    pub cost_comparison: SubgroupCostComparisonV1,
    pub subgroup_count: usize,
    pub policy_authorizes_subgroup_claim_only: bool,
    pub pruning_authorized: bool,
}

fn unresolved_pairs(certificate: &ExactStaticProbeCertificateV1) -> Vec<(String, String)> {
    certificate
        .intrinsically_unresolved_pairs
        .iter()
        .map(|pair| (pair.hypothesis_a.clone(), pair.hypothesis_b.clone()))
        .collect()
}

/// Compile a complete subgroup lattice, exact adaptive membership policy,
/// and exact static probe basis. The result is a candidate-identification
/// certificate only; pruning_authorized is always false.
pub fn compile_subgroup_identification(
    problem: &SubgroupIdentificationProblemV1,
    budget: SubgroupBudgetV1,
) -> Result<SubgroupIdentificationCertificateV1, SubgroupErrorV1> {
    validate_problem(problem, budget)?;
    let enumeration = enumerate_subgroups(&problem.group, budget)?;
    let active_problem = build_active_problem(problem, &enumeration);
    let planner_budget = QualificationBudgetV1 {
        maximum_hypotheses: budget.maximum_subgroups.min(MAX_SUBGROUPS),
        maximum_probes: budget.maximum_probes.min(MAX_PROBES),
        maximum_work_units: budget.maximum_work_units,
    };
    let static_budget = StaticProbeBudgetV1 {
        maximum_admissible_probes: 20,
        maximum_subsets: budget.maximum_static_subsets,
    };
    let static_probe_certificate = solve_exact_static_probe_design(
        &active_problem,
        planner_budget,
        static_budget,
    )
    .map_err(SubgroupErrorV1::Planner)?;

    let adaptive_policy = match compile_active_experiment(&active_problem, planner_budget) {
        Ok(certificate) => AdaptivePolicyV1::Complete(certificate),
        Err(ExperimentErrorV1::NoCompleteDiscriminationPolicy) => AdaptivePolicyV1::Incomplete {
            unresolved_pairs: unresolved_pairs(&static_probe_certificate),
        },
        Err(error) => return Err(SubgroupErrorV1::Planner(error)),
    };
    let cost_comparison = build_cost_comparison(
        problem,
        &active_problem,
        &adaptive_policy,
        &static_probe_certificate,
    )?;

    Ok(SubgroupIdentificationCertificateV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        problem_contract_bytes: problem.canonical_bytes(),
        subgroup_count: enumeration.subgroups.len(),
        enumeration,
        active_problem,
        adaptive_policy,
        static_probe_certificate,
        cost_comparison,
        policy_authorizes_subgroup_claim_only: true,
        pruning_authorized: false,
    })
}

pub fn verify_subgroup_identification_certificate(
    problem: &SubgroupIdentificationProblemV1,
    certificate: &SubgroupIdentificationCertificateV1,
    budget: SubgroupBudgetV1,
) -> Result<(), SubgroupErrorV1> {
    let replayed = compile_subgroup_identification(problem, budget)?;
    if *certificate != replayed {
        return Err(SubgroupErrorV1::ReceiptMismatch);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MembershipOutcomeV1 {
    Preserved,
    Violated,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MembershipObservationV1 {
    pub observation_id: String,
    pub probe_id: String,
    pub outcome: MembershipOutcomeV1,
    pub task_contract_id: String,
    pub checker_id: String,
    pub evidence_reference: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupFrontierSnapshotV1 {
    pub after_observation_count: usize,
    pub candidate_subgroup_ids: Vec<String>,
    pub model_consistent: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentificationStatusV1 {
    Unresolved,
    IdentifiedCandidate,
    ModelInconsistent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupObservationReceiptV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub problem_contract_bytes: Vec<u8>,
    pub observations_contract_bytes: Vec<u8>,
    pub snapshots: Vec<SubgroupFrontierSnapshotV1>,
    pub final_candidate_subgroup_ids: Vec<String>,
    pub status: IdentificationStatusV1,
    pub pruning_authorized: bool,
}

fn observations_bytes(observations: &[MembershipObservationV1]) -> Vec<u8> {
    let mut bytes = b"luminous.symm_subgroup_active.observations.v1\0".to_vec();
    append_u64(&mut bytes, observations.len() as u64);
    for observation in observations {
        append_string(&mut bytes, &observation.observation_id);
        append_string(&mut bytes, &observation.probe_id);
        bytes.push(match observation.outcome {
            MembershipOutcomeV1::Preserved => 1,
            MembershipOutcomeV1::Violated => 0,
            MembershipOutcomeV1::Unknown => 2,
        });
        append_string(&mut bytes, &observation.task_contract_id);
        append_string(&mut bytes, &observation.checker_id);
        append_string(&mut bytes, &observation.evidence_reference);
    }
    bytes
}

/// Replay membership evidence in any order. Unknown observations are retained
/// in the receipt but do not filter the subgroup frontier. Conflicting
/// repeated outcomes or a complete non-subgroup pattern produce an explicit
/// model-inconsistency status; the code never snaps to a nearby subgroup.
pub fn replay_membership_observations(
    problem: &SubgroupIdentificationProblemV1,
    certificate: &SubgroupIdentificationCertificateV1,
    observations: &[MembershipObservationV1],
    budget: SubgroupBudgetV1,
) -> Result<SubgroupObservationReceiptV1, SubgroupErrorV1> {
    verify_subgroup_identification_certificate(problem, certificate, budget)?;
    let mut seen_observation_ids = BTreeSet::new();
    let mut resolved_observations = Vec::with_capacity(observations.len());
    let mut results_by_probe: BTreeMap<usize, bool> = BTreeMap::new();
    let mut snapshots = Vec::with_capacity(observations.len() + 1);

    // Validate the entire batch before replay. An early inconsistency must
    // not hide a malformed or wrong-context observation later in the batch.
    for observation in observations {
        if !nonempty(&observation.observation_id) {
            return Err(SubgroupErrorV1::InvalidObservationId);
        }
        if !seen_observation_ids.insert(&observation.observation_id) {
            return Err(SubgroupErrorV1::DuplicateObservationId);
        }
        let probe_index = problem
            .probes
            .iter()
            .position(|probe| probe.probe_id == observation.probe_id)
            .ok_or_else(|| SubgroupErrorV1::UnknownProbe {
                probe_id: observation.probe_id.clone(),
            })?;
        let probe = &problem.probes[probe_index];
        if !probe.admissible {
            return Err(SubgroupErrorV1::InadmissibleProbe {
                probe_id: probe.probe_id.clone(),
            });
        }
        if observation.task_contract_id != probe.task_contract_id
            || observation.checker_id != probe.checker_id
        {
            return Err(SubgroupErrorV1::EvidenceContextMismatch {
                probe_id: probe.probe_id.clone(),
            });
        }
        if !nonempty(&observation.evidence_reference) {
            return Err(SubgroupErrorV1::MissingEvidenceReference {
                probe_id: probe.probe_id.clone(),
            });
        }
        let exact_outcome = match observation.outcome {
            MembershipOutcomeV1::Unknown => None,
            MembershipOutcomeV1::Preserved => Some(true),
            MembershipOutcomeV1::Violated => Some(false),
        };
        resolved_observations.push((probe_index, exact_outcome));
    }

    let candidate_ids = |results: &BTreeMap<usize, bool>| {
        certificate
            .enumeration
            .subgroups
            .iter()
            .filter(|subgroup| {
                results.iter().all(|(probe_index, preserved)| {
                    let element = problem.probes[*probe_index].group_element_index;
                    let predicted = subgroup.element_mask & (1_u64 << element) != 0;
                    predicted == *preserved
                })
            })
            .map(|subgroup| subgroup.subgroup_id.clone())
            .collect::<Vec<_>>()
    };

    let initial = candidate_ids(&results_by_probe);
    snapshots.push(SubgroupFrontierSnapshotV1 {
        after_observation_count: 0,
        candidate_subgroup_ids: initial.clone(),
        model_consistent: !initial.is_empty(),
    });

    for (observation_index, (probe_index, exact_outcome)) in
        resolved_observations.iter().enumerate()
    {
        if let Some(preserved) = exact_outcome {
            if let Some(previous) = results_by_probe.get(probe_index) {
                if previous != preserved {
                    snapshots.push(SubgroupFrontierSnapshotV1 {
                        after_observation_count: observation_index + 1,
                        candidate_subgroup_ids: Vec::new(),
                        model_consistent: false,
                    });
                    return Ok(SubgroupObservationReceiptV1 {
                        schema_version: 1,
                        problem_id: problem.problem_id.clone(),
                        problem_contract_bytes: problem.canonical_bytes(),
                        observations_contract_bytes: observations_bytes(observations),
                        snapshots,
                        final_candidate_subgroup_ids: Vec::new(),
                        status: IdentificationStatusV1::ModelInconsistent,
                        pruning_authorized: false,
                    });
                }
            } else {
                results_by_probe.insert(*probe_index, *preserved);
            }
        }

        let candidates = candidate_ids(&results_by_probe);
        let model_consistent = !candidates.is_empty();
        snapshots.push(SubgroupFrontierSnapshotV1 {
            after_observation_count: observation_index + 1,
            candidate_subgroup_ids: candidates.clone(),
            model_consistent,
        });
        if !model_consistent {
            return Ok(SubgroupObservationReceiptV1 {
                schema_version: 1,
                problem_id: problem.problem_id.clone(),
                problem_contract_bytes: problem.canonical_bytes(),
                observations_contract_bytes: observations_bytes(observations),
                snapshots,
                final_candidate_subgroup_ids: candidates,
                status: IdentificationStatusV1::ModelInconsistent,
                pruning_authorized: false,
            });
        }
    }

    let final_candidates = snapshots
        .last()
        .map(|snapshot| snapshot.candidate_subgroup_ids.clone())
        .unwrap_or_default();
    let status = if final_candidates.len() == 1 {
        IdentificationStatusV1::IdentifiedCandidate
    } else {
        IdentificationStatusV1::Unresolved
    };
    Ok(SubgroupObservationReceiptV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        problem_contract_bytes: problem.canonical_bytes(),
        observations_contract_bytes: observations_bytes(observations),
        snapshots,
        final_candidate_subgroup_ids: final_candidates,
        status,
        pruning_authorized: false,
    })
}

pub fn verify_membership_observation_receipt(
    problem: &SubgroupIdentificationProblemV1,
    certificate: &SubgroupIdentificationCertificateV1,
    observations: &[MembershipObservationV1],
    receipt: &SubgroupObservationReceiptV1,
    budget: SubgroupBudgetV1,
) -> Result<(), SubgroupErrorV1> {
    if receipt.schema_version != 1
        || receipt.problem_id != problem.problem_id
        || receipt.problem_contract_bytes != problem.canonical_bytes()
        || receipt.observations_contract_bytes != observations_bytes(observations)
    {
        return Err(SubgroupErrorV1::ReceiptMismatch);
    }
    let replayed = replay_membership_observations(problem, certificate, observations, budget)?;
    if *receipt != replayed {
        return Err(SubgroupErrorV1::ReceiptMismatch);
    }
    Ok(())
}

/// Recompile an exact adaptive policy on the current candidate frontier.
pub fn recommend_next_membership_probe(
    problem: &SubgroupIdentificationProblemV1,
    certificate: &SubgroupIdentificationCertificateV1,
    candidate_subgroup_ids: &[String],
    budget: SubgroupBudgetV1,
) -> Result<Option<(String, u64)>, SubgroupErrorV1> {
    verify_subgroup_identification_certificate(problem, certificate, budget)?;
    if candidate_subgroup_ids.is_empty() {
        return Err(SubgroupErrorV1::NoCandidateSubgroups);
    }
    let candidate_set: BTreeSet<&str> =
        candidate_subgroup_ids.iter().map(String::as_str).collect();
    if candidate_set.len() != candidate_subgroup_ids.len() {
        return Err(SubgroupErrorV1::InvalidCandidateFrontier);
    }
    let hypotheses = certificate
        .enumeration
        .subgroups
        .iter()
        .filter(|subgroup| candidate_set.contains(subgroup.subgroup_id.as_str()))
        .collect::<Vec<_>>();
    if hypotheses.len() != candidate_set.len() {
        return Err(SubgroupErrorV1::InvalidCandidateFrontier);
    }
    if hypotheses.len() <= 1 {
        return Ok(None);
    }

    let reduced = ActiveExperimentProblemV1 {
        problem_id: format!("{}::frontier-replay", problem.problem_id),
        hypotheses: hypotheses
            .iter()
            .map(|subgroup| subgroup.subgroup_id.clone())
            .collect(),
        probes: problem
            .probes
            .iter()
            .map(|probe| ExperimentProbeV1 {
                probe_id: probe.probe_id.clone(),
                cost: probe.cost,
                admissible: probe.admissible,
                outcomes: hypotheses
                    .iter()
                    .map(|subgroup| {
                        u32::from(
                            subgroup.element_mask & (1_u64 << probe.group_element_index) != 0,
                        )
                    })
                    .collect(),
            })
            .collect(),
    };
    let planner_budget = QualificationBudgetV1 {
        maximum_hypotheses: budget.maximum_subgroups.min(MAX_SUBGROUPS),
        maximum_probes: budget.maximum_probes.min(MAX_PROBES),
        maximum_work_units: budget.maximum_work_units,
    };
    match compile_active_experiment(&reduced, planner_budget) {
        Ok(plan) => match plan.root.disposition {
            PlanDispositionV1::Probe {
                probe_id,
                probe_cost,
                ..
            } => Ok(Some((probe_id, probe_cost))),
            PlanDispositionV1::Identified { .. } => Ok(None),
        },
        Err(ExperimentErrorV1::NoCompleteDiscriminationPolicy) => Ok(None),
        Err(error) => Err(SubgroupErrorV1::Planner(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cyclic_group(order: usize, group_id: &str) -> FiniteGroupTableV1 {
        let element_ids = (0..order).map(|i| format!("g{i}")).collect::<Vec<_>>();
        let multiplication = (0..order)
            .map(|left| (0..order).map(|right| (left + right) % order).collect())
            .collect();
        FiniteGroupTableV1 {
            group_id: group_id.to_owned(),
            element_ids,
            identity_index: 0,
            multiplication,
        }
    }

    fn v4_group() -> FiniteGroupTableV1 {
        let multiplication = (0..4)
            .map(|left| (0..4).map(|right| left ^ right).collect())
            .collect();
        FiniteGroupTableV1 {
            group_id: "V4".to_owned(),
            element_ids: vec!["e".into(), "a".into(), "b".into(), "ab".into()],
            identity_index: 0,
            multiplication,
        }
    }

    fn s3_group() -> FiniteGroupTableV1 {
        let permutations = [
            [0_usize, 1, 2],
            [1, 0, 2],
            [0, 2, 1],
            [2, 1, 0],
            [1, 2, 0],
            [2, 0, 1],
        ];
        let compose = |left: &[usize; 3], right: &[usize; 3]| -> [usize; 3] {
            [left[right[0]], left[right[1]], left[right[2]]]
        };
        let mut multiplication = vec![vec![0; 6]; 6];
        for left in 0..6 {
            for right in 0..6 {
                let product = compose(&permutations[left], &permutations[right]);
                multiplication[left][right] =
                    permutations.iter().position(|p| *p == product).unwrap();
            }
        }
        FiniteGroupTableV1 {
            group_id: "S3".to_owned(),
            element_ids: vec![
                "e".into(),
                "(01)".into(),
                "(12)".into(),
                "(02)".into(),
                "(012)".into(),
                "(021)".into(),
            ],
            identity_index: 0,
            multiplication,
        }
    }

    fn d4_group() -> FiniteGroupTableV1 {
        // Elements r^k s^e with r^4=e, s^2=e, s r s=r^-1.
        let mut multiplication = vec![vec![0; 8]; 8];
        for left in 0..8 {
            let k = left % 4;
            let e = left / 4;
            for right in 0..8 {
                let l = right % 4;
                let f = right / 4;
                let signed_l = if e == 0 { l as isize } else { -(l as isize) };
                let rotation = (k as isize + signed_l).rem_euclid(4) as usize;
                multiplication[left][right] = rotation + 4 * (e ^ f);
            }
        }
        FiniteGroupTableV1 {
            group_id: "D4".to_owned(),
            element_ids: vec![
                "e".into(), "r".into(), "r2".into(), "r3".into(),
                "s".into(), "rs".into(), "r2s".into(), "r3s".into(),
            ],
            identity_index: 0,
            multiplication,
        }
    }

    fn probes_for_group(group: &FiniteGroupTableV1, cost: u64) -> Vec<MembershipProbeV1> {
        (0..group.element_ids.len())
            .filter(|index| *index != group.identity_index)
            .map(|index| {
                let probe_cost = if group.group_id == "D4" {
                    // r, r2, r3, s, rs, r2s, r3s: total full-scan cost 11.
                    [1_u64, 1, 1, 2, 2, 2, 3][index - 1]
                } else {
                    cost
                };
                MembershipProbeV1 {
                    probe_id: format!("member-{}", group.element_ids[index]),
                    group_element_index: index,
                    task_contract_id: "task-contract-sha256:fixture".to_owned(),
                    checker_id: "finite-membership-checker-v1".to_owned(),
                    cost: probe_cost,
                    admissible: true,
                }
            })
            .collect()
    }

    fn problem(group: FiniteGroupTableV1) -> SubgroupIdentificationProblemV1 {
        let probes = probes_for_group(&group, 1);
        SubgroupIdentificationProblemV1 {
            problem_id: format!("active-subgroup-{}", group.group_id),
            group,
            probes,
        }
    }

    #[test]
    fn subgroup_lattice_counts_match_independent_c4_v4_s3_d4_oracles() {
        let cases = [
            (cyclic_group(4, "C4"), 3_usize),
            (v4_group(), 5),
            (s3_group(), 6),
            (d4_group(), 10),
        ];
        for (group, expected_count) in cases {
            let enumeration = enumerate_subgroups(&group, SubgroupBudgetV1::default()).unwrap();
            assert_eq!(enumeration.subgroups.len(), expected_count);
            assert!(enumeration.evidence.complete_subset_lattice_search);
            assert_eq!(
                enumeration.evidence.raw_subset_masks_scanned,
                1_u64 << group.element_ids.len()
            );
            assert_eq!(
                enumeration.evidence.identity_membership_checks,
                1_u64 << group.element_ids.len()
            );
            assert_eq!(enumeration.subgroups[0].element_indices, vec![0]);
            assert_eq!(
                enumeration.subgroups.last().unwrap().element_indices.len(),
                group.element_ids.len()
            );
        }
    }

    #[test]
    fn c4_and_d4_cost_targets_match_exact_adaptive_and_static_oracles() {
        let c4_problem = problem(cyclic_group(4, "C4"));
        let c4 = compile_subgroup_identification(&c4_problem, SubgroupBudgetV1::default()).unwrap();
        assert_eq!(c4.cost_comparison.full_scan_cost, 3);
        assert_eq!(c4.cost_comparison.adaptive_worst_case_cost, Some(2));

        let d4_problem = problem(d4_group());
        let d4 = compile_subgroup_identification(&d4_problem, SubgroupBudgetV1::default()).unwrap();
        assert_eq!(d4.cost_comparison.full_scan_cost, 11);
        assert_eq!(d4.cost_comparison.adaptive_worst_case_cost, Some(9));
        assert_eq!(d4.cost_comparison.static_basis_cost, 10);
        assert_eq!(d4.cost_comparison.uniform_prior_hypothesis_count, 10);
    }

    #[test]
    fn d4_membership_compiles_adaptive_and_static_policies_with_exact_replay() {
        let problem = problem(d4_group());
        let budget = SubgroupBudgetV1::default();
        let certificate = compile_subgroup_identification(&problem, budget).unwrap();
        assert_eq!(certificate.subgroup_count, 10);
        assert!(matches!(certificate.adaptive_policy, AdaptivePolicyV1::Complete(_)));
        assert!(certificate.static_probe_certificate.all_hypotheses_pairwise_distinguished);
        assert!(!certificate.pruning_authorized);
        verify_subgroup_identification_certificate(&problem, &certificate, budget).unwrap();
    }

    #[test]
    fn unknown_observations_never_remove_candidates_and_next_probe_is_replanned() {
        let problem = problem(cyclic_group(4, "C4"));
        let budget = SubgroupBudgetV1::default();
        let certificate = compile_subgroup_identification(&problem, budget).unwrap();
        let unknown = MembershipObservationV1 {
            observation_id: "obs-unknown".into(),
            probe_id: "member-g2".into(),
            outcome: MembershipOutcomeV1::Unknown,
            task_contract_id: "task-contract-sha256:fixture".into(),
            checker_id: "finite-membership-checker-v1".into(),
            evidence_reference: "evidence:unknown-reason-1".into(),
        };
        let receipt =
            replay_membership_observations(&problem, &certificate, &[unknown], budget).unwrap();
        assert_eq!(
            receipt.snapshots[0].candidate_subgroup_ids,
            receipt.snapshots[1].candidate_subgroup_ids
        );
        assert_eq!(receipt.final_candidate_subgroup_ids.len(), 3);
        assert_eq!(receipt.status, IdentificationStatusV1::Unresolved);
        assert!(!receipt.pruning_authorized);
        let recommendation = recommend_next_membership_probe(
            &problem,
            &certificate,
            &receipt.final_candidate_subgroup_ids,
            budget,
        )
        .unwrap();
        assert!(recommendation.is_some());
    }

    #[test]
    fn exact_membership_evidence_identifies_a_candidate_subgroup() {
        let problem = problem(cyclic_group(4, "C4"));
        let budget = SubgroupBudgetV1::default();
        let certificate = compile_subgroup_identification(&problem, budget).unwrap();
        let observations = vec![
            MembershipObservationV1 {
                observation_id: "obs-r".into(),
                probe_id: "member-g1".into(),
                outcome: MembershipOutcomeV1::Violated,
                task_contract_id: "task-contract-sha256:fixture".into(),
                checker_id: "finite-membership-checker-v1".into(),
                evidence_reference: "counterexample:case-r".into(),
            },
            MembershipObservationV1 {
                observation_id: "obs-r2".into(),
                probe_id: "member-g2".into(),
                outcome: MembershipOutcomeV1::Preserved,
                task_contract_id: "task-contract-sha256:fixture".into(),
                checker_id: "finite-membership-checker-v1".into(),
                evidence_reference: "proof:case-r2".into(),
            },
        ];
        let receipt =
            replay_membership_observations(&problem, &certificate, &observations, budget).unwrap();
        assert_eq!(receipt.status, IdentificationStatusV1::IdentifiedCandidate);
        assert_eq!(receipt.final_candidate_subgroup_ids.len(), 1);
        verify_membership_observation_receipt(
            &problem, &certificate, &observations, &receipt, budget
        )
        .unwrap();
    }

    #[test]
    fn non_subgroup_preservation_pattern_is_model_inconsistency() {
        let problem = problem(d4_group());
        let budget = SubgroupBudgetV1::default();
        let certificate = compile_subgroup_identification(&problem, budget).unwrap();
        // {e, r, s} is not closed: r*s = rs is absent.
        let preserved = BTreeSet::from([0_usize, 1_usize, 4_usize]);
        let observations = problem.probes.iter().map(|probe| {
            MembershipObservationV1 {
                observation_id: format!("obs-{}", probe.probe_id),
                probe_id: probe.probe_id.clone(),
                outcome: if preserved.contains(&probe.group_element_index) {
                    MembershipOutcomeV1::Preserved
                } else {
                    MembershipOutcomeV1::Violated
                },
                task_contract_id: probe.task_contract_id.clone(),
                checker_id: probe.checker_id.clone(),
                evidence_reference: format!("evidence:{}", probe.probe_id),
            }
        }).collect::<Vec<_>>();
        let receipt =
            replay_membership_observations(&problem, &certificate, &observations, budget).unwrap();
        assert_eq!(receipt.status, IdentificationStatusV1::ModelInconsistent);
        assert!(receipt.final_candidate_subgroup_ids.is_empty());
        assert!(!receipt.pruning_authorized);
    }

    #[test]
    fn inconsistent_duplicate_results_and_wrong_evidence_context_fail_closed() {
        let problem = problem(cyclic_group(4, "C4"));
        let budget = SubgroupBudgetV1::default();
        let certificate = compile_subgroup_identification(&problem, budget).unwrap();
        let duplicate = vec![
            MembershipObservationV1 {
                observation_id: "obs-1".into(),
                probe_id: "member-g1".into(),
                outcome: MembershipOutcomeV1::Preserved,
                task_contract_id: "task-contract-sha256:fixture".into(),
                checker_id: "finite-membership-checker-v1".into(),
                evidence_reference: "evidence:1".into(),
            },
            MembershipObservationV1 {
                observation_id: "obs-2".into(),
                probe_id: "member-g1".into(),
                outcome: MembershipOutcomeV1::Violated,
                task_contract_id: "task-contract-sha256:fixture".into(),
                checker_id: "finite-membership-checker-v1".into(),
                evidence_reference: "evidence:2".into(),
            },
        ];
        let receipt =
            replay_membership_observations(&problem, &certificate, &duplicate, budget).unwrap();
        assert_eq!(receipt.status, IdentificationStatusV1::ModelInconsistent);

        let mut wrong_context = duplicate[0].clone();
        wrong_context.task_contract_id = "different-contract".into();
        assert_eq!(
            replay_membership_observations(
                &problem, &certificate, &[wrong_context], budget
            ),
            Err(SubgroupErrorV1::EvidenceContextMismatch {
                probe_id: "member-g1".into()
            })
        );
    }

    #[test]
    fn non_group_tables_and_work_exhaustion_are_rejected() {
        let mut invalid = cyclic_group(4, "bad-c4");
        invalid.multiplication[1][1] = 0;
        assert_eq!(
            enumerate_subgroups(&invalid, SubgroupBudgetV1::default()),
            Err(SubgroupErrorV1::IdentityLawViolation)
        );
        let low_budget = SubgroupBudgetV1 {
            maximum_work_units: 1,
            ..SubgroupBudgetV1::default()
        };
        assert!(matches!(
            enumerate_subgroups(&d4_group(), low_budget),
            Err(SubgroupErrorV1::WorkBudgetExceeded { .. })
        ));
    }

    #[test]
    fn stale_certificate_and_inadmissible_probe_are_rejected() {
        let mut problem = problem(cyclic_group(4, "C4"));
        let budget = SubgroupBudgetV1::default();
        let mut certificate = compile_subgroup_identification(&problem, budget).unwrap();
        certificate.pruning_authorized = true;
        assert_eq!(
            verify_subgroup_identification_certificate(&problem, &certificate, budget),
            Err(SubgroupErrorV1::ReceiptMismatch)
        );

        let original = compile_subgroup_identification(&problem, budget).unwrap();
        problem.probes[0].admissible = false;
        let forged_observation = MembershipObservationV1 {
            observation_id: "obs-inadmissible".into(),
            probe_id: "member-g0".into(),
            outcome: MembershipOutcomeV1::Preserved,
            task_contract_id: "task-contract-sha256:fixture".into(),
            checker_id: "finite-membership-checker-v1".into(),
            evidence_reference: "evidence:bad".into(),
        };
        assert_eq!(
            replay_membership_observations(
                &problem, &original, &[forged_observation], budget
            ),
            Err(SubgroupErrorV1::ReceiptMismatch)
        );
    }
}
