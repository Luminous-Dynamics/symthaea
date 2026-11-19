// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Active identification of the task-preserving subgroup of a known finite
//! group. Candidate hypotheses are exact subgroups, never arbitrary subsets.
//! The resulting certificate is identification evidence only, not pruning
//! authority.

use std::collections::BTreeSet;
use std::fmt;

use symmetry_active_experiment_reference::{
    ActiveExperimentCertificateV1, ActiveExperimentProblemV1, ExactStaticProbeCertificateV1,
    ExperimentErrorV1, ExperimentProbeV1, PlanDispositionV1, PlanNodeV1,
    QualificationBudgetV1, StaticProbeBudgetV1, compile_active_experiment,
    replay_observations, solve_exact_static_probe_design,
    verify_active_experiment_certificate,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteGroupContractV1 {
    pub group_id: String,
    pub element_ids: Vec<String>,
    /// Complete multiplication table: multiplication[a][b] is the index of a*b.
    pub multiplication: Vec<Vec<usize>>,
    pub identity: usize,
}

impl FiniteGroupContractV1 {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"luminous.symm_subgroup_active.group.v1\0".to_vec();
        append_string(&mut bytes, &self.group_id);
        append_u64(&mut bytes, self.element_ids.len() as u64);
        for element_id in &self.element_ids {
            append_string(&mut bytes, element_id);
        }
        append_u64(&mut bytes, self.identity as u64);
        for row in &self.multiplication {
            append_u64(&mut bytes, row.len() as u64);
            for product in row {
                append_u64(&mut bytes, *product as u64);
            }
        }
        bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubgroupEnumerationBudgetV1 {
    pub maximum_group_order: usize,
    pub maximum_subgroups: usize,
    pub maximum_work_units: u64,
}

impl Default for SubgroupEnumerationBudgetV1 {
    fn default() -> Self {
        Self {
            maximum_group_order: 16,
            maximum_subgroups: 16,
            maximum_work_units: 20_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupHypothesisV1 {
    pub subgroup_id: String,
    pub element_indices: Vec<usize>,
    pub element_ids: Vec<String>,
    pub bit_mask: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupCatalogueV1 {
    pub schema_version: u32,
    pub group_id: String,
    pub group_contract_bytes: Vec<u8>,
    pub group_order: usize,
    pub subgroups: Vec<SubgroupHypothesisV1>,
    pub subgroup_masks_examined: u64,
    pub closure_checks: u64,
    pub work_units: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MembershipProbeV1 {
    pub probe_id: String,
    pub group_element_index: usize,
    pub cost: u64,
    pub admissible: bool,
    /// Identity of the independent task-membership checker expected to answer.
    pub checker_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupIdentificationProblemV1 {
    pub problem_id: String,
    pub context_id: String,
    pub task_contract_id: String,
    pub group: FiniteGroupContractV1,
    pub probes: Vec<MembershipProbeV1>,
}

impl SubgroupIdentificationProblemV1 {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"luminous.symm_subgroup_active.problem.v1\0".to_vec();
        append_string(&mut bytes, &self.problem_id);
        append_string(&mut bytes, &self.context_id);
        append_string(&mut bytes, &self.task_contract_id);
        append_bytes(&mut bytes, &self.group.canonical_bytes());
        append_u64(&mut bytes, self.probes.len() as u64);
        for probe in &self.probes {
            append_string(&mut bytes, &probe.probe_id);
            append_u64(&mut bytes, probe.group_element_index as u64);
            append_u64(&mut bytes, probe.cost);
            bytes.push(u8::from(probe.admissible));
            append_string(&mut bytes, &probe.checker_id);
        }
        bytes
    }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubgroupErrorV1 {
    InvalidIdentifier,
    EmptyGroup,
    GroupOrderExceeded { required: usize, maximum: usize },
    MultiplicationTableNotSquare,
    ProductOutOfRange,
    MultiplicationTableNotLatin,
    NonAssociativeGroup,
    InvalidIdentity,
    MissingInverse,
    DuplicateElementId,
    ArithmeticOverflow,
    WorkBudgetExceeded { required: u64, maximum: u64 },
    TooManySubgroups { required: usize, maximum: usize },
    TooManyProbes { required: usize, maximum: usize },
    DuplicateProbeId,
    DuplicateGroupElementProbe,
    InvalidProbeElement,
    ZeroProbeCost,
    TooManySubgroupHypotheses,
    NoDiscriminatingProbeFamily,
    InvalidObservationOrder,
    UnknownProbe,
    InvalidEvidenceReference,
    ReceiptMismatch,
    Planner(ExperimentErrorV1),
}

impl fmt::Display for SubgroupErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SubgroupErrorV1 {}

fn add_work(work: &mut u64, amount: u64) -> Result<(), SubgroupErrorV1> {
    *work = work.checked_add(amount).ok_or(SubgroupErrorV1::ArithmeticOverflow)?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ValidatedGroupV1 {
    inverses: Vec<usize>,
    work_units: u64,
}

fn validate_group(
    group: &FiniteGroupContractV1,
    budget: SubgroupEnumerationBudgetV1,
) -> Result<ValidatedGroupV1, SubgroupErrorV1> {
    let order = group.element_ids.len();
    if group.group_id.trim().is_empty() {
        return Err(SubgroupErrorV1::InvalidIdentifier);
    }
    if order == 0 {
        return Err(SubgroupErrorV1::EmptyGroup);
    }
    let maximum = budget.maximum_group_order.min(16);
    if order > maximum {
        return Err(SubgroupErrorV1::GroupOrderExceeded {
            required: order,
            maximum,
        });
    }
    if group.multiplication.len() != order
        || group.multiplication.iter().any(|row| row.len() != order)
    {
        return Err(SubgroupErrorV1::MultiplicationTableNotSquare);
    }
    if group.identity >= order {
        return Err(SubgroupErrorV1::InvalidIdentity);
    }
    let mut ids = BTreeSet::new();
    for id in &group.element_ids {
        if id.trim().is_empty() {
            return Err(SubgroupErrorV1::InvalidIdentifier);
        }
        if !ids.insert(id) {
            return Err(SubgroupErrorV1::DuplicateElementId);
        }
    }

    let n = u64::try_from(order).map_err(|_| SubgroupErrorV1::ArithmeticOverflow)?;
    let n2 = n.checked_mul(n).ok_or(SubgroupErrorV1::ArithmeticOverflow)?;
    let n3 = n2.checked_mul(n).ok_or(SubgroupErrorV1::ArithmeticOverflow)?;
    let masks = 1_u64.checked_shl(order as u32).ok_or(SubgroupErrorV1::ArithmeticOverflow)?;
    let closure_bound = masks.checked_mul(n2).ok_or(SubgroupErrorV1::ArithmeticOverflow)?;
    let required = n3
        .checked_add(n2.checked_mul(2).ok_or(SubgroupErrorV1::ArithmeticOverflow)?)
        .and_then(|v| v.checked_add(n.checked_mul(2)?))
        .and_then(|v| v.checked_add(closure_bound))
        .ok_or(SubgroupErrorV1::ArithmeticOverflow)?;
    if required > budget.maximum_work_units {
        return Err(SubgroupErrorV1::WorkBudgetExceeded {
            required,
            maximum: budget.maximum_work_units,
        });
    }

    let mut work_units = 0_u64;
    for row in 0..order {
        let mut seen = vec![false; order];
        for column in 0..order {
            add_work(&mut work_units, 1)?;
            let product = group.multiplication[row][column];
            if product >= order {
                return Err(SubgroupErrorV1::ProductOutOfRange);
            }
            if seen[product] {
                return Err(SubgroupErrorV1::MultiplicationTableNotLatin);
            }
            seen[product] = true;
        }
    }
    for column in 0..order {
        let mut seen = vec![false; order];
        for row in 0..order {
            add_work(&mut work_units, 1)?;
            let product = group.multiplication[row][column];
            if product >= order || seen[product] {
                return Err(SubgroupErrorV1::MultiplicationTableNotLatin);
            }
            seen[product] = true;
        }
    }
    for a in 0..order {
        for b in 0..order {
            for c in 0..order {
                add_work(&mut work_units, 1)?;
                if group.multiplication[group.multiplication[a][b]][c]
                    != group.multiplication[a][group.multiplication[b][c]]
                {
                    return Err(SubgroupErrorV1::NonAssociativeGroup);
                }
            }
        }
    }
    for element in 0..order {
        add_work(&mut work_units, 2)?;
        if group.multiplication[group.identity][element] != element
            || group.multiplication[element][group.identity] != element
        {
            return Err(SubgroupErrorV1::InvalidIdentity);
        }
    }

    let mut inverses = vec![usize::MAX; order];
    for element in 0..order {
        let mut found = None;
        for candidate in 0..order {
            add_work(&mut work_units, 1)?;
            if group.multiplication[element][candidate] == group.identity
                && group.multiplication[candidate][element] == group.identity
            {
                found = Some(candidate);
                break;
            }
        }
        inverses[element] = found.ok_or(SubgroupErrorV1::MissingInverse)?;
    }
    Ok(ValidatedGroupV1 { inverses, work_units })
}

/// Exhaustively enumerate every subgroup of an admitted group of order <= 16.
pub fn enumerate_subgroups(
    group: &FiniteGroupContractV1,
    budget: SubgroupEnumerationBudgetV1,
) -> Result<SubgroupCatalogueV1, SubgroupErrorV1> {
    let validated = validate_group(group, budget)?;
    let order = group.element_ids.len();
    let mask_count = 1_u64.checked_shl(order as u32).ok_or(SubgroupErrorV1::ArithmeticOverflow)?;
    let mut work_units = validated.work_units;
    let mut closure_checks = 0_u64;
    let mut subgroups = Vec::new();

    for mask in 0..mask_count {
        if mask & (1_u64 << group.identity) == 0 {
            continue;
        }
        let members: Vec<usize> = (0..order)
            .filter(|index| mask & (1_u64 << index) != 0)
            .collect();
        let mut is_subgroup = true;
        for left in &members {
            add_work(&mut work_units, 1)?;
            closure_checks = closure_checks.checked_add(1)
                .ok_or(SubgroupErrorV1::ArithmeticOverflow)?;
            if mask & (1_u64 << validated.inverses[*left]) == 0 {
                is_subgroup = false;
                break;
            }
            for right in &members {
                add_work(&mut work_units, 1)?;
                closure_checks = closure_checks.checked_add(1)
                    .ok_or(SubgroupErrorV1::ArithmeticOverflow)?;
                let product = group.multiplication[*left][*right];
                if mask & (1_u64 << product) == 0 {
                    is_subgroup = false;
                    break;
                }
            }
            if !is_subgroup {
                break;
            }
        }
        if !is_subgroup {
            continue;
        }
        let element_ids: Vec<String> = members.iter()
            .map(|index| group.element_ids[*index].clone())
            .collect();
        let subgroup_id = format!(
            "H[{}]",
            members.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
        );
        subgroups.push(SubgroupHypothesisV1 {
            subgroup_id,
            element_indices: members,
            element_ids,
            bit_mask: mask,
        });
        let maximum = budget.maximum_subgroups.min(16);
        if subgroups.len() > maximum {
            return Err(SubgroupErrorV1::TooManySubgroups {
                required: subgroups.len(),
                maximum,
            });
        }
    }
    subgroups.sort_by(|left, right| {
        (left.element_indices.len(), &left.element_indices)
            .cmp(&(right.element_indices.len(), &right.element_indices))
    });
    Ok(SubgroupCatalogueV1 {
        schema_version: 1,
        group_id: group.group_id.clone(),
        group_contract_bytes: group.canonical_bytes(),
        group_order: order,
        subgroups,
        subgroup_masks_examined: mask_count,
        closure_checks,
        work_units,
    })
}

fn validate_problem(
    problem: &SubgroupIdentificationProblemV1,
    budget: SubgroupEnumerationBudgetV1,
) -> Result<SubgroupCatalogueV1, SubgroupErrorV1> {
    if problem.problem_id.trim().is_empty()
        || problem.context_id.trim().is_empty()
        || problem.task_contract_id.trim().is_empty()
    {
        return Err(SubgroupErrorV1::InvalidIdentifier);
    }
    let catalogue = enumerate_subgroups(&problem.group, budget)?;
    if problem.probes.len() > 64 {
        return Err(SubgroupErrorV1::TooManyProbes {
            required: problem.probes.len(),
            maximum: 64,
        });
    }
    let mut probe_ids = BTreeSet::new();
    let mut elements = BTreeSet::new();
    for probe in &problem.probes {
        if probe.probe_id.trim().is_empty() || probe.checker_id.trim().is_empty() {
            return Err(SubgroupErrorV1::InvalidIdentifier);
        }
        if !probe_ids.insert(&probe.probe_id) {
            return Err(SubgroupErrorV1::DuplicateProbeId);
        }
        if !elements.insert(probe.group_element_index) {
            return Err(SubgroupErrorV1::DuplicateGroupElementProbe);
        }
        if probe.group_element_index >= catalogue.group_order {
            return Err(SubgroupErrorV1::InvalidProbeElement);
        }
        if probe.cost == 0 {
            return Err(SubgroupErrorV1::ZeroProbeCost);
        }
    }
    Ok(catalogue)
}

fn make_active_problem(
    problem: &SubgroupIdentificationProblemV1,
    catalogue: &SubgroupCatalogueV1,
) -> ActiveExperimentProblemV1 {
    ActiveExperimentProblemV1 {
        problem_id: format!("{}::{}", problem.problem_id, problem.context_id),
        hypotheses: catalogue.subgroups.iter()
            .map(|subgroup| subgroup.subgroup_id.clone())
            .collect(),
        probes: problem.probes.iter().map(|probe| ExperimentProbeV1 {
            probe_id: probe.probe_id.clone(),
            cost: probe.cost,
            admissible: probe.admissible,
            outcomes: catalogue.subgroups.iter().map(|subgroup| {
                u32::from(subgroup.element_indices.contains(&probe.group_element_index))
            }).collect(),
        }).collect(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnresolvedSubgroupPairV1 {
    pub subgroup_a_id: String,
    pub subgroup_b_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnresolvedSubgroupFrontierV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub context_id: String,
    pub problem_contract_bytes: Vec<u8>,
    pub subgroup_catalogue: SubgroupCatalogueV1,
    pub unresolved_pairs: Vec<UnresolvedSubgroupPairV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StaticBaselineV1 {
    Exact(ExactStaticProbeCertificateV1),
    SkippedForBudget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupPolicyCertificateV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub context_id: String,
    pub task_contract_id: String,
    pub group_contract_bytes: Vec<u8>,
    pub problem_contract_bytes: Vec<u8>,
    pub subgroup_catalogue: SubgroupCatalogueV1,
    pub active_problem: ActiveExperimentProblemV1,
    pub adaptive_policy: ActiveExperimentCertificateV1,
    pub static_baseline: StaticBaselineV1,
    pub pruning_authority: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubgroupPolicyOutcomeV1 {
    Identifiable(SubgroupPolicyCertificateV1),
    Unresolved(UnresolvedSubgroupFrontierV1),
}

fn unresolved_pairs(problem: &ActiveExperimentProblemV1) -> Vec<UnresolvedSubgroupPairV1> {
    let mut result = Vec::new();
    for a in 0..problem.hypotheses.len() {
        for b in (a + 1)..problem.hypotheses.len() {
            let separated = problem.probes.iter().any(|probe| {
                probe.admissible && probe.outcomes[a] != probe.outcomes[b]
            });
            if !separated {
                result.push(UnresolvedSubgroupPairV1 {
                    subgroup_a_id: problem.hypotheses[a].clone(),
                    subgroup_b_id: problem.hypotheses[b].clone(),
                });
            }
        }
    }
    result
}

/// Compile an exact adaptive membership policy over the subgroup lattice.
/// This function never returns task-pruning authority.
pub fn compile_subgroup_policy(
    problem: &SubgroupIdentificationProblemV1,
    enum_budget: SubgroupEnumerationBudgetV1,
    active_budget: QualificationBudgetV1,
    static_budget: StaticProbeBudgetV1,
) -> Result<SubgroupPolicyOutcomeV1, SubgroupErrorV1> {
    let catalogue = validate_problem(problem, enum_budget)?;
    if catalogue.subgroups.is_empty() {
        return Err(SubgroupErrorV1::NoDiscriminatingProbeFamily);
    }
    if catalogue.subgroups.len() > active_budget.maximum_hypotheses.min(16) {
        return Err(SubgroupErrorV1::TooManySubgroupHypotheses);
    }
    let active_problem = make_active_problem(problem, &catalogue);
    let frontier = unresolved_pairs(&active_problem);
    if !frontier.is_empty() {
        return Ok(SubgroupPolicyOutcomeV1::Unresolved(UnresolvedSubgroupFrontierV1 {
            schema_version: 1,
            problem_id: problem.problem_id.clone(),
            context_id: problem.context_id.clone(),
            problem_contract_bytes: problem.canonical_bytes(),
            subgroup_catalogue: catalogue,
            unresolved_pairs: frontier,
        }));
    }

    let adaptive_policy = compile_active_experiment(&active_problem, active_budget)
        .map_err(SubgroupErrorV1::Planner)?;
    verify_active_experiment_certificate(&active_problem, &adaptive_policy, active_budget)
        .map_err(SubgroupErrorV1::Planner)?;

    let admissible_count = active_problem.probes.iter().filter(|probe| probe.admissible).count();
    let subset_count = if admissible_count <= 63 {
        1_u64.checked_shl(admissible_count as u32).unwrap_or(u64::MAX)
    } else {
        u64::MAX
    };
    let static_baseline = if admissible_count <= usize::from(static_budget.maximum_admissible_probes)
        && subset_count <= static_budget.maximum_subsets
    {
        StaticBaselineV1::Exact(
            solve_exact_static_probe_design(&active_problem, active_budget, static_budget)
                .map_err(SubgroupErrorV1::Planner)?,
        )
    } else {
        StaticBaselineV1::SkippedForBudget
    };

    Ok(SubgroupPolicyOutcomeV1::Identifiable(SubgroupPolicyCertificateV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        context_id: problem.context_id.clone(),
        task_contract_id: problem.task_contract_id.clone(),
        group_contract_bytes: problem.group.canonical_bytes(),
        problem_contract_bytes: problem.canonical_bytes(),
        subgroup_catalogue: catalogue,
        active_problem,
        adaptive_policy,
        static_baseline,
        pruning_authority: "none",
    }))
}

pub fn verify_subgroup_policy_certificate(
    problem: &SubgroupIdentificationProblemV1,
    certificate: &SubgroupPolicyCertificateV1,
    enum_budget: SubgroupEnumerationBudgetV1,
    active_budget: QualificationBudgetV1,
    static_budget: StaticProbeBudgetV1,
) -> Result<(), SubgroupErrorV1> {
    match compile_subgroup_policy(problem, enum_budget, active_budget, static_budget)? {
        SubgroupPolicyOutcomeV1::Identifiable(expected) if expected == *certificate => Ok(()),
        _ => Err(SubgroupErrorV1::ReceiptMismatch),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MembershipEvidenceV1 {
    Preserved {
        checker_id: String,
        task_contract_id: String,
        evidence_reference: String,
    },
    Violated {
        checker_id: String,
        task_contract_id: String,
        environment_id: String,
        counterexample_reference: String,
        differing_semantic_fields: Vec<String>,
    },
    Unknown {
        checker_id: String,
        task_contract_id: String,
        reason_reference: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MembershipObservationV1 {
    pub probe_id: String,
    pub evidence: MembershipEvidenceV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubgroupReplayStatusV1 {
    IdentifiedCandidate { subgroup_id: String },
    Ambiguous { next_probe_id: Option<String> },
    PausedUnknown { probe_id: String },
    Inconsistent { probe_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupReplayReceiptV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub context_id: String,
    pub policy_contract_bytes: Vec<u8>,
    pub observations: Vec<MembershipObservationV1>,
    pub remaining_subgroup_ids: Vec<String>,
    pub status: SubgroupReplayStatusV1,
    pub checked_cost: u64,
    pub pruning_authority: &'static str,
}

fn evidence_outcome(
    problem: &SubgroupIdentificationProblemV1,
    probe: &MembershipProbeV1,
    evidence: &MembershipEvidenceV1,
) -> Result<Option<bool>, SubgroupErrorV1> {
    let (checker_id, task_id) = match evidence {
        MembershipEvidenceV1::Preserved {
            checker_id, task_contract_id, evidence_reference,
        } => {
            if evidence_reference.trim().is_empty() {
                return Err(SubgroupErrorV1::InvalidEvidenceReference);
            }
            (checker_id, task_contract_id)
        }
        MembershipEvidenceV1::Violated {
            checker_id, task_contract_id, environment_id,
            counterexample_reference, differing_semantic_fields,
        } => {
            let unique: BTreeSet<&String> = differing_semantic_fields.iter().collect();
            if environment_id != &problem.context_id
                || counterexample_reference.trim().is_empty()
                || differing_semantic_fields.is_empty()
                || differing_semantic_fields.iter().any(|field| field.trim().is_empty())
                || unique.len() != differing_semantic_fields.len()
            {
                return Err(SubgroupErrorV1::InvalidEvidenceReference);
            }
            (checker_id, task_contract_id)
        }
        MembershipEvidenceV1::Unknown {
            checker_id, task_contract_id, reason_reference,
        } => {
            if reason_reference.trim().is_empty() {
                return Err(SubgroupErrorV1::InvalidEvidenceReference);
            }
            (checker_id, task_contract_id)
        }
    };
    if checker_id != &probe.checker_id || task_id != &problem.task_contract_id {
        return Err(SubgroupErrorV1::InvalidEvidenceReference);
    }
    Ok(match evidence {
        MembershipEvidenceV1::Preserved { .. } => Some(true),
        MembershipEvidenceV1::Violated { .. } => Some(false),
        MembershipEvidenceV1::Unknown { .. } => None,
    })
}

fn subgroup_ids_for_mask(catalogue: &SubgroupCatalogueV1, mask: u64) -> Vec<String> {
    catalogue.subgroups.iter().enumerate()
        .filter(|(index, _)| mask & (1_u64 << index) != 0)
        .map(|(_, subgroup)| subgroup.subgroup_id.clone())
        .collect()
}

fn next_probe(node: &PlanNodeV1) -> Option<String> {
    match &node.disposition {
        PlanDispositionV1::Identified { .. } => None,
        PlanDispositionV1::Probe { probe_id, .. } => Some(probe_id.clone()),
    }
}

/// Replay checker observations through the adaptive policy. Unknown pauses
/// without eliminating candidates. A missing branch means that the observed
/// preservation pattern is not any subgroup of the supplied ambient group.
pub fn replay_subgroup_membership(
    problem: &SubgroupIdentificationProblemV1,
    certificate: &SubgroupPolicyCertificateV1,
    observations: &[MembershipObservationV1],
    enum_budget: SubgroupEnumerationBudgetV1,
    active_budget: QualificationBudgetV1,
    static_budget: StaticProbeBudgetV1,
) -> Result<SubgroupReplayReceiptV1, SubgroupErrorV1> {
    verify_subgroup_policy_certificate(
        problem, certificate, enum_budget, active_budget, static_budget,
    )?;
    let mut node = &certificate.adaptive_policy.root;
    let mut outcomes = Vec::new();
    let mut checked_cost = 0_u64;

    for (observation_index, observation) in observations.iter().enumerate() {
        let (probe_index, expected_probe_id, branches) = match &node.disposition {
            PlanDispositionV1::Identified { .. } => return Err(SubgroupErrorV1::InvalidObservationOrder),
            PlanDispositionV1::Probe { probe_index, probe_id, branches, .. } => {
                (*probe_index, probe_id, branches)
            }
        };
        if observation.probe_id != *expected_probe_id {
            return Err(SubgroupErrorV1::InvalidObservationOrder);
        }
        let probe = problem.probes.get(probe_index).ok_or(SubgroupErrorV1::UnknownProbe)?;
        checked_cost = checked_cost.checked_add(probe.cost)
            .ok_or(SubgroupErrorV1::ArithmeticOverflow)?;

        let Some(preserved) = evidence_outcome(problem, probe, &observation.evidence)? else {
            if observation_index + 1 != observations.len() {
                return Err(SubgroupErrorV1::InvalidObservationOrder);
            }
            return Ok(SubgroupReplayReceiptV1 {
                schema_version: 1,
                problem_id: problem.problem_id.clone(),
                context_id: problem.context_id.clone(),
                policy_contract_bytes: problem.canonical_bytes(),
                observations: observations.to_vec(),
                remaining_subgroup_ids: subgroup_ids_for_mask(
                    &certificate.subgroup_catalogue, node.hypotheses_mask,
                ),
                status: SubgroupReplayStatusV1::PausedUnknown {
                    probe_id: probe.probe_id.clone(),
                },
                checked_cost,
                pruning_authority: "none",
            });
        };

        let outcome = u32::from(preserved);
        let Some(branch) = branches.iter().find(|branch| branch.outcome == outcome) else {
            if observation_index + 1 != observations.len() {
                return Err(SubgroupErrorV1::InvalidObservationOrder);
            }
            return Ok(SubgroupReplayReceiptV1 {
                schema_version: 1,
                problem_id: problem.problem_id.clone(),
                context_id: problem.context_id.clone(),
                policy_contract_bytes: problem.canonical_bytes(),
                observations: observations.to_vec(),
                remaining_subgroup_ids: Vec::new(),
                status: SubgroupReplayStatusV1::Inconsistent {
                    probe_id: probe.probe_id.clone(),
                },
                checked_cost,
                pruning_authority: "none",
            });
        };
        outcomes.push(outcome);
        node = &branch.child;
    }

    let remaining = subgroup_ids_for_mask(&certificate.subgroup_catalogue, node.hypotheses_mask);
    let status = match &node.disposition {
        PlanDispositionV1::Identified { hypothesis_id, .. } => {
            SubgroupReplayStatusV1::IdentifiedCandidate {
                subgroup_id: hypothesis_id.clone(),
            }
        }
        PlanDispositionV1::Probe { .. } if remaining.len() == 1 => {
            SubgroupReplayStatusV1::IdentifiedCandidate {
                subgroup_id: remaining[0].clone(),
            }
        }
        PlanDispositionV1::Probe { .. } => SubgroupReplayStatusV1::Ambiguous {
            next_probe_id: next_probe(node),
        },
    };
    let replayed = replay_observations(
        &certificate.active_problem, &certificate.adaptive_policy, &outcomes, active_budget,
    ).map_err(SubgroupErrorV1::Planner)?;
    if replayed != remaining {
        return Err(SubgroupErrorV1::ReceiptMismatch);
    }
    Ok(SubgroupReplayReceiptV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        context_id: problem.context_id.clone(),
        policy_contract_bytes: problem.canonical_bytes(),
        observations: observations.to_vec(),
        remaining_subgroup_ids: remaining,
        status,
        checked_cost,
        pruning_authority: "none",
    })
}

pub fn verify_subgroup_replay_receipt(
    problem: &SubgroupIdentificationProblemV1,
    certificate: &SubgroupPolicyCertificateV1,
    receipt: &SubgroupReplayReceiptV1,
    enum_budget: SubgroupEnumerationBudgetV1,
    active_budget: QualificationBudgetV1,
    static_budget: StaticProbeBudgetV1,
) -> Result<(), SubgroupErrorV1> {
    let replayed = replay_subgroup_membership(
        problem, certificate, &receipt.observations,
        enum_budget, active_budget, static_budget,
    )?;
    if replayed == *receipt {
        Ok(())
    } else {
        Err(SubgroupErrorV1::ReceiptMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cyclic_group(order: usize, id: &str) -> FiniteGroupContractV1 {
        FiniteGroupContractV1 {
            group_id: id.to_owned(),
            element_ids: (0..order).map(|index| format!("r{index}")).collect(),
            multiplication: (0..order).map(|a| {
                (0..order).map(|b| (a + b) % order).collect()
            }).collect(),
            identity: 0,
        }
    }

    fn dihedral_group(rotation_order: usize, id: &str) -> FiniteGroupContractV1 {
        let order = 2 * rotation_order;
        let mut multiplication = vec![vec![0; order]; order];
        for a in 0..order {
            let rotation_a = a % rotation_order;
            let reflection_a = a / rotation_order;
            for b in 0..order {
                let rotation_b = b % rotation_order;
                let reflection_b = b / rotation_order;
                let signed_rotation_b = if reflection_a == 0 {
                    rotation_b
                } else {
                    (rotation_order - rotation_b) % rotation_order
                };
                let rotation = (rotation_a + signed_rotation_b) % rotation_order;
                let reflection = (reflection_a + reflection_b) % 2;
                multiplication[a][b] = rotation + rotation_order * reflection;
            }
        }
        let mut element_ids: Vec<String> = (0..rotation_order)
            .map(|index| format!("r{index}")).collect();
        element_ids.extend((0..rotation_order).map(|index| format!("r{index}s")));
        FiniteGroupContractV1 {
            group_id: id.to_owned(),
            element_ids,
            multiplication,
            identity: 0,
        }
    }

    fn problem(group: FiniteGroupContractV1, probe_indices: Vec<usize>) -> SubgroupIdentificationProblemV1 {
        let probes = probe_indices.into_iter().map(|index| MembershipProbeV1 {
            probe_id: format!("member:{}", group.element_ids[index]),
            group_element_index: index,
            cost: if index <= 3 { 1 } else { 2 },
            admissible: true,
            checker_id: "exact-task-membership-v1".to_owned(),
        }).collect();
        SubgroupIdentificationProblemV1 {
            problem_id: format!("{}-active-subgroups-v1", group.group_id),
            context_id: "fixture-environment-v1".to_owned(),
            task_contract_id: "fixture-task-contract-v1".to_owned(),
            group,
            probes,
        }
    }

    fn budgets() -> (SubgroupEnumerationBudgetV1, QualificationBudgetV1, StaticProbeBudgetV1) {
        (
            SubgroupEnumerationBudgetV1::default(),
            QualificationBudgetV1::default(),
            StaticProbeBudgetV1::default(),
        )
    }

    fn evidence(problem: &SubgroupIdentificationProblemV1, probe: &MembershipProbeV1, result: bool)
        -> MembershipEvidenceV1
    {
        if result {
            MembershipEvidenceV1::Preserved {
                checker_id: probe.checker_id.clone(),
                task_contract_id: problem.task_contract_id.clone(),
                evidence_reference: format!("fixture-proof:{}", probe.probe_id),
            }
        } else {
            MembershipEvidenceV1::Violated {
                checker_id: probe.checker_id.clone(),
                task_contract_id: problem.task_contract_id.clone(),
                environment_id: problem.context_id.clone(),
                counterexample_reference: format!("fixture-counterexample:{}", probe.probe_id),
                differing_semantic_fields: vec!["boundary-condition".to_owned()],
            }
        }
    }

    #[test]
    fn exact_subgroup_lattice_counts_match_small_group_oracles() {
        for (group, expected) in [
            (cyclic_group(4, "C4"), 3),
            (dihedral_group(2, "V4"), 5),
            (dihedral_group(3, "S3"), 6),
            (dihedral_group(4, "D4"), 10),
        ] {
            let catalogue = enumerate_subgroups(&group, SubgroupEnumerationBudgetV1::default()).unwrap();
            assert_eq!(catalogue.subgroups.len(), expected, "{}", group.group_id);
            for subgroup in &catalogue.subgroups {
                assert!(subgroup.element_indices.contains(&group.identity));
                for a in &subgroup.element_indices {
                    for b in &subgroup.element_indices {
                        assert!(subgroup.element_indices.contains(&group.multiplication[*a][*b]));
                    }
                }
            }
        }
    }

    #[test]
    fn adaptive_policy_identifies_every_d4_subgroup_and_replays() {
        let p = problem(dihedral_group(4, "D4"), (1..8).collect());
        let (eb, ab, sb) = budgets();
        let SubgroupPolicyOutcomeV1::Identifiable(cert) = compile_subgroup_policy(&p, eb, ab, sb).unwrap()
        else { panic!("complete D4 membership probes should distinguish all subgroups"); };
        assert_eq!(cert.subgroup_catalogue.subgroups.len(), 10);
        assert_eq!(cert.pruning_authority, "none");

        let full_scan_cost: u64 = p.probes.iter().filter(|probe| probe.admissible).map(|probe| probe.cost).sum();
        assert!(cert.adaptive_policy.root_worst_case_cost <= full_scan_cost);
        assert!(matches!(cert.static_baseline, StaticBaselineV1::Exact(_)));
        verify_subgroup_policy_certificate(&p, &cert, eb, ab, sb).unwrap();

        for target in &cert.subgroup_catalogue.subgroups {
            let mut node = &cert.adaptive_policy.root;
            let mut records = Vec::new();
            loop {
                match &node.disposition {
                    PlanDispositionV1::Identified { .. } => break,
                    PlanDispositionV1::Probe { probe_index, branches, .. } => {
                        let probe = &p.probes[*probe_index];
                        let preserves = target.element_indices.contains(&probe.group_element_index);
                        let outcome = u32::from(preserves);
                        records.push(MembershipObservationV1 {
                            probe_id: probe.probe_id.clone(),
                            evidence: evidence(&p, probe, preserves),
                        });
                        node = &branches.iter().find(|branch| branch.outcome == outcome)
                            .expect("subgroup outcome has a policy branch").child;
                    }
                }
            }
            let receipt = replay_subgroup_membership(&p, &cert, &records, eb, ab, sb).unwrap();
            assert_eq!(receipt.remaining_subgroup_ids, vec![target.subgroup_id.clone()]);
            assert_eq!(receipt.status, SubgroupReplayStatusV1::IdentifiedCandidate {
                subgroup_id: target.subgroup_id.clone()
            });
            verify_subgroup_replay_receipt(&p, &cert, &receipt, eb, ab, sb).unwrap();
        }
    }

    #[test]
    fn frozen_cost_benchmarks_report_c4_and_d4_against_full_scan() {
        let (eb, ab, sb) = budgets();

        let c4_problem = problem(cyclic_group(4, "C4"), vec![1, 2, 3]);
        let SubgroupPolicyOutcomeV1::Identifiable(c4) =
            compile_subgroup_policy(&c4_problem, eb, ab, sb).unwrap()
        else {
            panic!("complete C4 membership family should be identifiable");
        };
        let c4_full_scan: u64 = c4_problem.probes.iter()
            .filter(|probe| probe.admissible)
            .map(|probe| probe.cost)
            .sum();
        assert_eq!(c4_full_scan, 3);
        assert_eq!(c4.adaptive_policy.root_worst_case_cost, 2);
        assert_eq!(2 * c4_full_scan, 3 * c4.adaptive_policy.root_worst_case_cost);

        let d4_problem = problem(dihedral_group(4, "D4"), (1..8).collect());
        let SubgroupPolicyOutcomeV1::Identifiable(d4) =
            compile_subgroup_policy(&d4_problem, eb, ab, sb).unwrap()
        else {
            panic!("complete D4 membership family should be identifiable");
        };
        let d4_full_scan: u64 = d4_problem.probes.iter()
            .filter(|probe| probe.admissible)
            .map(|probe| probe.cost)
            .sum();
        assert_eq!(d4_full_scan, 11);
        assert_eq!(d4.adaptive_policy.root_worst_case_cost, 9);
        let StaticBaselineV1::Exact(static_basis) = d4.static_baseline else {
            panic!("D4 static basis fits the exact subset budget");
        };
        assert_eq!(static_basis.selected_cost, 10);
        assert!(static_basis.optimality_certified);
    }

    #[test]
    fn incomplete_probe_family_returns_unresolved_pairs() {
        let p = problem(dihedral_group(4, "D4"), vec![1]);
        let (eb, ab, sb) = budgets();
        let outcome = compile_subgroup_policy(&p, eb, ab, sb).unwrap();
        let SubgroupPolicyOutcomeV1::Unresolved(frontier) = outcome else {
            panic!("single element probe cannot distinguish D4's complete subgroup lattice");
        };
        assert!(!frontier.unresolved_pairs.is_empty());
    }

    #[test]
    fn unknown_observation_pauses_and_keeps_all_candidates() {
        let p = problem(dihedral_group(4, "D4"), (1..8).collect());
        let (eb, ab, sb) = budgets();
        let SubgroupPolicyOutcomeV1::Identifiable(cert) = compile_subgroup_policy(&p, eb, ab, sb).unwrap()
        else { panic!("D4 probe family is complete"); };
        let PlanDispositionV1::Probe { probe_index, .. } = &cert.adaptive_policy.root.disposition
        else { panic!("policy root must be a membership query"); };
        let probe = &p.probes[*probe_index];
        let record = MembershipObservationV1 {
            probe_id: probe.probe_id.clone(),
            evidence: MembershipEvidenceV1::Unknown {
                checker_id: probe.checker_id.clone(),
                task_contract_id: p.task_contract_id.clone(),
                reason_reference: "fixture-unavailable-check".to_owned(),
            },
        };
        let receipt = replay_subgroup_membership(&p, &cert, &[record], eb, ab, sb).unwrap();
        assert!(matches!(receipt.status, SubgroupReplayStatusV1::PausedUnknown { .. }));
        assert_eq!(receipt.remaining_subgroup_ids.len(), 10);
        assert_eq!(receipt.pruning_authority, "none");
    }

    #[test]
    fn non_subgroup_preservation_pattern_is_model_inconsistent() {
        let p = problem(dihedral_group(4, "D4"), (1..8).collect());
        let (eb, ab, sb) = budgets();
        let SubgroupPolicyOutcomeV1::Identifiable(cert) = compile_subgroup_policy(&p, eb, ab, sb).unwrap()
        else { panic!("complete D4 probe family is required"); };
        // {e, r, s} is not a subgroup because r*s is excluded.
        let claimed = BTreeSet::from([0_usize, 1, 4]);
        let mut node = &cert.adaptive_policy.root;
        let mut records = Vec::new();
        let mut inconsistent = false;
        loop {
            match &node.disposition {
                PlanDispositionV1::Identified { .. } => break,
                PlanDispositionV1::Probe { probe_index, branches, .. } => {
                    let probe = &p.probes[*probe_index];
                    let preserves = claimed.contains(&probe.group_element_index);
                    let outcome = u32::from(preserves);
                    records.push(MembershipObservationV1 {
                        probe_id: probe.probe_id.clone(),
                        evidence: evidence(&p, probe, preserves),
                    });
                    if let Some(branch) = branches.iter().find(|branch| branch.outcome == outcome) {
                        node = &branch.child;
                    } else {
                        let receipt = replay_subgroup_membership(&p, &cert, &records, eb, ab, sb).unwrap();
                        assert_eq!(receipt.status, SubgroupReplayStatusV1::Inconsistent {
                            probe_id: probe.probe_id.clone()
                        });
                        assert!(receipt.remaining_subgroup_ids.is_empty());
                        inconsistent = true;
                        break;
                    }
                }
            }
        }
        assert!(inconsistent);
    }

    #[test]
    fn malformed_group_tables_and_budgets_fail_closed() {
        let mut malformed = dihedral_group(4, "D4");
        malformed.multiplication[1][1] = usize::MAX;
        assert_eq!(
            enumerate_subgroups(&malformed, SubgroupEnumerationBudgetV1::default()),
            Err(SubgroupErrorV1::ProductOutOfRange)
        );
        let tiny = SubgroupEnumerationBudgetV1 {
            maximum_work_units: 1,
            ..SubgroupEnumerationBudgetV1::default()
        };
        assert!(matches!(
            enumerate_subgroups(&dihedral_group(4, "D4"), tiny),
            Err(SubgroupErrorV1::WorkBudgetExceeded { .. })
        ));
    }

    #[test]
    fn evidence_is_bound_to_checker_and_task_contract() {
        let p = problem(dihedral_group(4, "D4"), (1..8).collect());
        let (eb, ab, sb) = budgets();
        let SubgroupPolicyOutcomeV1::Identifiable(cert) = compile_subgroup_policy(&p, eb, ab, sb).unwrap()
        else { panic!("complete D4 probe family is required"); };
        let PlanDispositionV1::Probe { probe_index, .. } = &cert.adaptive_policy.root.disposition
        else { panic!("policy root must be a probe"); };
        let probe = &p.probes[*probe_index];
        let bad = MembershipObservationV1 {
            probe_id: probe.probe_id.clone(),
            evidence: MembershipEvidenceV1::Preserved {
                checker_id: "wrong-checker".to_owned(),
                task_contract_id: p.task_contract_id.clone(),
                evidence_reference: "fixture-proof".to_owned(),
            },
        };
        assert_eq!(
            replay_subgroup_membership(&p, &cert, &[bad], eb, ab, sb),
            Err(SubgroupErrorV1::InvalidEvidenceReference)
        );
    }

    #[test]
    fn changed_task_contract_invalidates_policy_certificate() {
        let p = problem(dihedral_group(4, "D4"), (1..8).collect());
        let (eb, ab, sb) = budgets();
        let SubgroupPolicyOutcomeV1::Identifiable(cert) = compile_subgroup_policy(&p, eb, ab, sb).unwrap()
        else { panic!("complete D4 probe family is required"); };
        let mut changed = p.clone();
        changed.task_contract_id = "mutated-task-contract".to_owned();
        assert_eq!(
            verify_subgroup_policy_certificate(&changed, &cert, eb, ab, sb),
            Err(SubgroupErrorV1::ReceiptMismatch)
        );
    }
}
