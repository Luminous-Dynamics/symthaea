// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact finite-group quotient construction for bounded discrete search tasks.
//! Discovery is not proof; task preservation is a separate fail-closed obligation.

// Explicit indices keep finite multiplication/action tables auditable.
#![allow(clippy::needless_range_loop)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteGroupV1 {
    pub group_id: String,
    pub identity: usize,
    pub multiplication: Vec<Vec<usize>>,
}

/// Left-action convention: p_g(p_h(x)) = p_(g*h)(x).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteActionV1 {
    pub action_id: String,
    pub group: FiniteGroupV1,
    pub permutations: Vec<Vec<usize>>,
}

/// Exact task fields required to be invariant before pruning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateSemanticsV1 {
    pub in_search_domain: bool,
    pub admissible: bool,
    pub feasible: bool,
    pub objective_cost: i64,
    pub constraint_signature: Vec<i64>,
    pub boundary_condition_id: String,
    pub output_equivalence_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchProblemV1 {
    pub problem_id: String,
    pub candidate_ids: Vec<String>,
    pub semantics: Vec<CandidateSemanticsV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualificationBudgetV1 {
    pub maximum_group_order: usize,
    pub maximum_candidates: usize,
    pub maximum_work_units: usize,
}

impl Default for QualificationBudgetV1 {
    fn default() -> Self {
        Self {
            maximum_group_order: 64,
            maximum_candidates: 10_000,
            maximum_work_units: 50_000_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruningAuthorityV1 {
    ExactFiniteGroupAndExactTaskSemanticPreservation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemberWitnessV1 {
    pub candidate_index: usize,
    pub candidate_id: String,
    pub group_element: usize,
    pub inverse_group_element: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrbitV1 {
    pub representative_index: usize,
    pub representative_id: String,
    pub members: Vec<MemberWitnessV1>,
    /// Complete stabilizer subgroup: every g with g·representative = representative.
    pub stabilizer_group_elements: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CertificateEvidenceV1 {
    pub group_axiom_checks: usize,
    pub action_law_checks: usize,
    pub task_semantic_checks: usize,
    pub orbit_member_witnesses: usize,
    pub stabilizer_membership_checks: usize,
    pub stabilizer_subgroup_closure_checks: usize,
    pub stabilizer_witnesses: usize,
    pub orbit_stabilizer_checks: usize,
    pub complete_domain_cover: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuotientCertificateV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub action_id: String,
    pub group_id: String,
    pub group_order: usize,
    pub candidate_count: usize,
    pub pruning_authority: PruningAuthorityV1,
    pub orbits: Vec<OrbitV1>,
    pub evidence: CertificateEvidenceV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskInvarianceViolationV1 {
    pub group_element: usize,
    pub source_index: usize,
    pub image_index: usize,
    pub source_candidate_id: String,
    pub image_candidate_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileOutcomeV1 {
    Authorized(QuotientCertificateV1),
    Blocked(TaskInvarianceViolationV1),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationReceiptV1 {
    pub problem_id: String,
    pub action_id: String,
    pub group_id: String,
    pub group_order: usize,
    pub candidate_count: usize,
    pub orbit_count: usize,
    pub witnessed_member_count: usize,
    pub group_axiom_checks: usize,
    pub action_law_checks: usize,
    pub task_semantic_checks: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiftedCandidateV1 {
    pub candidate_index: usize,
    pub candidate_id: String,
    pub group_element_from_representative: usize,
    pub inverse_group_element: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuotientErrorV1 {
    InvalidIdentifier,
    EmptyGroup,
    EmptyCandidateUniverse,
    GroupOrderExceeded,
    CandidateCountExceeded,
    WorkBudgetExceeded,
    ArithmeticOverflow,
    MultiplicationTableNotSquare,
    ProductOutOfRange,
    MultiplicationTableNotLatin,
    NonAssociativeGroup,
    InvalidIdentity,
    MissingInverse,
    CandidateSemanticsLengthMismatch,
    DuplicateCandidateId,
    ActionCountMismatch,
    PermutationLengthMismatch,
    NotPermutation,
    IdentityActionMismatch,
    ActionLawViolation,
    CertificateIdentityMismatch,
    WrongPruningAuthority,
    EmptyOrbit,
    CandidateIndexOutOfRange,
    CandidateIdMismatch,
    BadRepresentative,
    BadWitness,
    BadInverseWitness,
    BadStabilizerWitness,
    StabilizerNotSubgroup,
    OrbitStabilizerMismatch,
    TaskSemanticsMismatch,
    IncompleteOrbit,
    OverlappingOrbit,
    IncompleteDomainCover,
    NonCanonicalOrdering,
    RequestedCandidateIsNotRepresentative,
    EvidenceCountMismatch,
}

impl fmt::Display for QuotientErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for QuotientErrorV1 {}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ValidatedGroupV1 {
    inverses: Vec<usize>,
    checks: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ValidatedActionV1 {
    group: ValidatedGroupV1,
    checks: usize,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn ceil_log2(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

/// Conservative loop/comparison budget for group, action, task, and orbit work.
/// It includes BTreeSet/sort comparison envelopes, not allocator or wall-clock cost.
fn preflight(
    group_order: usize,
    candidate_count: usize,
    budget: QualificationBudgetV1,
) -> Result<(), QuotientErrorV1> {
    if group_order == 0 {
        return Err(QuotientErrorV1::EmptyGroup);
    }
    if group_order > budget.maximum_group_order {
        return Err(QuotientErrorV1::GroupOrderExceeded);
    }
    if candidate_count == 0 {
        return Err(QuotientErrorV1::EmptyCandidateUniverse);
    }
    if candidate_count > budget.maximum_candidates {
        return Err(QuotientErrorV1::CandidateCountExceeded);
    }

    let g2 = group_order
        .checked_mul(group_order)
        .ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    let g3 = g2
        .checked_mul(group_order)
        .ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    let gn = group_order
        .checked_mul(candidate_count)
        .ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    let g2n = g2
        .checked_mul(candidate_count)
        .ok_or(QuotientErrorV1::ArithmeticOverflow)?;

    // Group: Latin rows/columns (2g²), inverse search (<=g²),
    // associativity (g³), and two-sided identity (2g).
    let group_work = g3
        .checked_add(
            g2.checked_mul(3)
                .ok_or(QuotientErrorV1::ArithmeticOverflow)?,
        )
        .and_then(|value| value.checked_add(group_order.checked_mul(2)?))
        .ok_or(QuotientErrorV1::ArithmeticOverflow)?;

    // Action: permutation validation (gn), identity action (n), and
    // composition law (g²n). Task preservation then checks every g*x (gn).
    let action_work = g2n
        .checked_add(gn)
        .and_then(|value| value.checked_add(candidate_count))
        .ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    let task_work = gn;

    // Orbit construction: generator visits, witness lookup, and ordered-set
    // insertion are bounded by 3gn plus the corresponding logarithmic
    // comparison envelopes for candidate/orbit ordering.
    let ordering_work = candidate_count
        .checked_mul(ceil_log2(candidate_count))
        .and_then(|value| value.checked_add(gn.checked_mul(ceil_log2(group_order))?))
        .ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    let orbit_work = gn
        .checked_mul(3)
        .and_then(|value| value.checked_add(ordering_work))
        .ok_or(QuotientErrorV1::ArithmeticOverflow)?;

    // Stabilizers: membership scan (gn), subgroup inverse/product closure
    // (at most g²n), and one orbit-stabilizer identity per candidate at most.
    let stabilizer_work = g2n
        .checked_add(gn)
        .and_then(|value| value.checked_add(candidate_count))
        .ok_or(QuotientErrorV1::ArithmeticOverflow)?;

    let required = group_work
        .checked_add(action_work)
        .and_then(|value| value.checked_add(task_work))
        .and_then(|value| value.checked_add(orbit_work))
        .and_then(|value| value.checked_add(stabilizer_work))
        .ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    if required > budget.maximum_work_units {
        return Err(QuotientErrorV1::WorkBudgetExceeded);
    }
    Ok(())
}

fn validate_group(
    group: &FiniteGroupV1,
    budget: QualificationBudgetV1,
) -> Result<ValidatedGroupV1, QuotientErrorV1> {
    if !nonempty(&group.group_id) {
        return Err(QuotientErrorV1::InvalidIdentifier);
    }
    let order = group.multiplication.len();
    if order == 0 {
        return Err(QuotientErrorV1::EmptyGroup);
    }
    if order > budget.maximum_group_order {
        return Err(QuotientErrorV1::GroupOrderExceeded);
    }
    if group.multiplication.iter().any(|row| row.len() != order) {
        return Err(QuotientErrorV1::MultiplicationTableNotSquare);
    }
    if group.identity >= order {
        return Err(QuotientErrorV1::InvalidIdentity);
    }

    let mut checks = 0usize;
    for row in 0..order {
        let mut seen = vec![false; order];
        for column in 0..order {
            checks += 1;
            let value = group.multiplication[row][column];
            if value >= order {
                return Err(QuotientErrorV1::ProductOutOfRange);
            }
            if seen[value] {
                return Err(QuotientErrorV1::MultiplicationTableNotLatin);
            }
            seen[value] = true;
        }
    }
    for column in 0..order {
        let mut seen = vec![false; order];
        for row in 0..order {
            checks += 1;
            let value = group.multiplication[row][column];
            if seen[value] {
                return Err(QuotientErrorV1::MultiplicationTableNotLatin);
            }
            seen[value] = true;
        }
    }
    for a in 0..order {
        for b in 0..order {
            for c in 0..order {
                checks += 1;
                if group.multiplication[group.multiplication[a][b]][c]
                    != group.multiplication[a][group.multiplication[b][c]]
                {
                    return Err(QuotientErrorV1::NonAssociativeGroup);
                }
            }
        }
    }
    for element in 0..order {
        checks += 2;
        if group.multiplication[group.identity][element] != element
            || group.multiplication[element][group.identity] != element
        {
            return Err(QuotientErrorV1::InvalidIdentity);
        }
    }

    let mut inverses = vec![usize::MAX; order];
    for element in 0..order {
        let mut inverse = None;
        for candidate in 0..order {
            checks += 1;
            if group.multiplication[element][candidate] == group.identity
                && group.multiplication[candidate][element] == group.identity
            {
                inverse = Some(candidate);
                break;
            }
        }
        inverses[element] = inverse.ok_or(QuotientErrorV1::MissingInverse)?;
    }
    Ok(ValidatedGroupV1 { inverses, checks })
}


/// Resource limits for the exhaustive finite subgroup lattice oracle.
///
/// The oracle is intended for small reference groups used to generate exact
/// benchmark labels. It never truncates a lattice: exceeding any budget is an
/// explicit error, not a partial list presented as complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubgroupLatticeBudgetV1 {
    pub maximum_group_order: usize,
    pub maximum_subgroups: usize,
    pub maximum_work_units: u64,
}

impl Default for SubgroupLatticeBudgetV1 {
    fn default() -> Self {
        Self {
            maximum_group_order: 64,
            maximum_subgroups: 65_536,
            maximum_work_units: 50_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubgroupLatticeErrorV1 {
    InvalidBudget,
    GroupOrderExceedsBitmask,
    SubgroupCountExceeded,
    WorkBudgetExceeded,
    ArithmeticOverflow,
    InvalidGroup(QuotientErrorV1),
    ReceiptMismatch,
}

impl fmt::Display for SubgroupLatticeErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SubgroupLatticeErrorV1 {}

/// One exact subgroup. Elements and generators are sorted by group index.
///
/// The generator list is a witness: closing it under the supplied group's
/// multiplication and inverse operations must reproduce the element list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupV1 {
    pub elements: Vec<usize>,
    pub generators: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupLatticeEvidenceV1 {
    pub group_axiom_checks: usize,
    pub generated_closure_calls: usize,
    /// Includes group checks, generated-closure work, containment and maximality scans.
    pub total_work_units: u64,
    pub containment_checks: u64,
    pub maximality_checks: u64,
    pub complete_lattice: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupLatticeReceiptV1 {
    pub schema_version: u32,
    pub group_id: String,
    pub group_order: usize,
    pub group_contract_bytes: Vec<u8>,
    pub budget: SubgroupLatticeBudgetV1,
    /// Sorted by subgroup order, then by the element bitmask.
    pub subgroups: Vec<SubgroupV1>,
    /// Indices into subgroups; each is maximal among proper subgroups.
    pub maximal_proper_subgroup_indices: Vec<usize>,
    pub evidence: SubgroupLatticeEvidenceV1,
}

fn append_subgroup_bytes(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(&(value.len() as u64).to_le_bytes());
    target.extend_from_slice(value);
}

fn finite_group_canonical_bytes(group: &FiniteGroupV1) -> Vec<u8> {
    let mut bytes = b"luminous.finite_group.v1\\0".to_vec();
    append_subgroup_bytes(&mut bytes, group.group_id.as_bytes());
    bytes.extend_from_slice(&(group.identity as u64).to_le_bytes());
    bytes.extend_from_slice(&(group.multiplication.len() as u64).to_le_bytes());
    for row in &group.multiplication {
        bytes.extend_from_slice(&(row.len() as u64).to_le_bytes());
        for product in row {
            bytes.extend_from_slice(&(*product as u64).to_le_bytes());
        }
    }
    bytes
}

fn spend_subgroup_work(
    work_units: &mut u64,
    maximum: u64,
    amount: u64,
) -> Result<(), SubgroupLatticeErrorV1> {
    *work_units = work_units
        .checked_add(amount)
        .ok_or(SubgroupLatticeErrorV1::ArithmeticOverflow)?;
    if *work_units > maximum {
        return Err(SubgroupLatticeErrorV1::WorkBudgetExceeded);
    }
    Ok(())
}

fn subgroup_mask_to_elements(mask: u64, group_order: usize) -> Vec<usize> {
    (0..group_order)
        .filter(|element| mask & (1_u64 << element) != 0)
        .collect()
}

/// Compute the closure of an existing subgroup plus one additional generator.
fn close_generated_subgroup(
    group: &FiniteGroupV1,
    inverses: &[usize],
    initial_mask: u64,
    generator: usize,
    work_units: &mut u64,
    maximum_work_units: u64,
) -> Result<u64, SubgroupLatticeErrorV1> {
    let order = group.multiplication.len();
    let mut mask = initial_mask | (1_u64 << generator) | (1_u64 << group.identity);
    loop {
        let previous = mask;
        for left in 0..order {
            if mask & (1_u64 << left) == 0 {
                continue;
            }
            spend_subgroup_work(work_units, maximum_work_units, 1)?;
            mask |= 1_u64 << inverses[left];
            for right in 0..order {
                if mask & (1_u64 << right) == 0 {
                    continue;
                }
                spend_subgroup_work(work_units, maximum_work_units, 1)?;
                mask |= 1_u64 << group.multiplication[left][right];
            }
        }
        if mask == previous {
            return Ok(mask);
        }
    }
}

/// Exhaustively enumerate every subgroup of a finite group of order at most 64.
///
/// Starting from the identity subgroup, repeatedly adjoin one element and
/// close under products and inverses. Every finite subgroup has a finite
/// generating set, so this reaches the complete lattice. A BTreeMap deduplicates
/// equal subgroups reached through different generating sets. All validation,
/// closure, containment and maximality work consumes an explicit budget.
pub fn enumerate_subgroup_lattice(
    group: &FiniteGroupV1,
    budget: SubgroupLatticeBudgetV1,
) -> Result<SubgroupLatticeReceiptV1, SubgroupLatticeErrorV1> {
    if budget.maximum_group_order == 0
        || budget.maximum_group_order > 64
        || budget.maximum_subgroups == 0
        || budget.maximum_work_units == 0
    {
        return Err(SubgroupLatticeErrorV1::InvalidBudget);
    }

    let order = group.multiplication.len();
    if order > 64 {
        return Err(SubgroupLatticeErrorV1::GroupOrderExceedsBitmask);
    }
    if order == 0 || order > budget.maximum_group_order {
        return Err(SubgroupLatticeErrorV1::InvalidGroup(
            if order == 0 {
                QuotientErrorV1::EmptyGroup
            } else {
                QuotientErrorV1::GroupOrderExceeded
            },
        ));
    }

    let validation_budget = QualificationBudgetV1 {
        maximum_group_order: budget.maximum_group_order,
        maximum_candidates: 1,
        maximum_work_units: usize::MAX,
    };
    let validated = validate_group(group, validation_budget)
        .map_err(SubgroupLatticeErrorV1::InvalidGroup)?;
    let mut total_work_units = u64::try_from(validated.checks)
        .map_err(|_| SubgroupLatticeErrorV1::ArithmeticOverflow)?;
    if total_work_units > budget.maximum_work_units {
        return Err(SubgroupLatticeErrorV1::WorkBudgetExceeded);
    }

    let full_mask = if order == 64 {
        u64::MAX
    } else {
        (1_u64 << order) - 1
    };
    let identity_mask = 1_u64 << group.identity;

    // Each map value is a deterministic generating witness for its subgroup.
    let mut known: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    let mut queue = VecDeque::new();
    known.insert(identity_mask, Vec::new());
    queue.push_back(identity_mask);

    let mut generated_closure_calls = 0usize;
    while let Some(current_mask) = queue.pop_front() {
        let current_generators = known
            .get(&current_mask)
            .cloned()
            .ok_or(SubgroupLatticeErrorV1::ReceiptMismatch)?;

        for generator in 0..order {
            if current_mask & (1_u64 << generator) != 0 {
                continue;
            }
            generated_closure_calls = generated_closure_calls
                .checked_add(1)
                .ok_or(SubgroupLatticeErrorV1::ArithmeticOverflow)?;
            let closure = close_generated_subgroup(
                group,
                &validated.inverses,
                current_mask,
                generator,
                &mut total_work_units,
                budget.maximum_work_units,
            )?;
            if known.contains_key(&closure) {
                continue;
            }
            if known.len() >= budget.maximum_subgroups {
                return Err(SubgroupLatticeErrorV1::SubgroupCountExceeded);
            }
            let mut generators = current_generators.clone();
            generators.push(generator);
            generators.sort_unstable();
            generators.dedup();
            known.insert(closure, generators);
            queue.push_back(closure);
        }
    }

    if !known.contains_key(&full_mask) {
        return Err(SubgroupLatticeErrorV1::ReceiptMismatch);
    }

    let mut ordered_masks: Vec<u64> = known.keys().copied().collect();
    ordered_masks.sort_by_key(|mask| (mask.count_ones(), *mask));
    let mut subgroups = Vec::with_capacity(ordered_masks.len());
    for mask in &ordered_masks {
        let generators = known
            .get(mask)
            .cloned()
            .ok_or(SubgroupLatticeErrorV1::ReceiptMismatch)?;
        subgroups.push(SubgroupV1 {
            elements: subgroup_mask_to_elements(*mask, order),
            generators,
        });
    }

    let mut maximal_proper_subgroup_indices = Vec::new();
    let mut containment_checks = 0u64;
    let mut maximality_checks = 0u64;
    for (index, subgroup_mask) in ordered_masks.iter().copied().enumerate() {
        if subgroup_mask == full_mask {
            continue;
        }
        let mut has_intermediate = false;
        for other_mask in ordered_masks.iter().copied() {
            if other_mask == subgroup_mask || other_mask == full_mask {
                continue;
            }
            spend_subgroup_work(&mut total_work_units, budget.maximum_work_units, 1)?;
            containment_checks = containment_checks
                .checked_add(1)
                .ok_or(SubgroupLatticeErrorV1::ArithmeticOverflow)?;
            let contains = subgroup_mask & !other_mask == 0;
            if contains {
                spend_subgroup_work(&mut total_work_units, budget.maximum_work_units, 1)?;
                maximality_checks = maximality_checks
                    .checked_add(1)
                    .ok_or(SubgroupLatticeErrorV1::ArithmeticOverflow)?;
                has_intermediate = true;
            }
        }
        if !has_intermediate {
            maximal_proper_subgroup_indices.push(index);
        }
    }

    Ok(SubgroupLatticeReceiptV1 {
        schema_version: 1,
        group_id: group.group_id.clone(),
        group_order: order,
        group_contract_bytes: finite_group_canonical_bytes(group),
        budget,
        subgroups,
        maximal_proper_subgroup_indices,
        evidence: SubgroupLatticeEvidenceV1 {
            group_axiom_checks: validated.checks,
            generated_closure_calls,
            total_work_units,
            containment_checks,
            maximality_checks,
            complete_lattice: true,
        },
    })
}

/// Recomputes the complete lattice, generator witnesses and maximal-subgroup
/// labels from the exact multiplication table and budget.
pub fn verify_subgroup_lattice_receipt(
    group: &FiniteGroupV1,
    receipt: &SubgroupLatticeReceiptV1,
) -> Result<(), SubgroupLatticeErrorV1> {
    if receipt.schema_version != 1
        || receipt.group_id != group.group_id
        || receipt.group_order != group.multiplication.len()
        || receipt.group_contract_bytes != finite_group_canonical_bytes(group)
        || !receipt.evidence.complete_lattice
    {
        return Err(SubgroupLatticeErrorV1::ReceiptMismatch);
    }
    let replayed = enumerate_subgroup_lattice(group, receipt.budget)?;
    if *receipt != replayed {
        return Err(SubgroupLatticeErrorV1::ReceiptMismatch);
    }

    let validation_budget = QualificationBudgetV1 {
        maximum_group_order: receipt.budget.maximum_group_order,
        maximum_candidates: 1,
        maximum_work_units: usize::MAX,
    };
    let validated = validate_group(group, validation_budget)
        .map_err(SubgroupLatticeErrorV1::InvalidGroup)?;
    for subgroup in &receipt.subgroups {
        let mut mask = 1_u64 << group.identity;
        let mut work_units = u64::try_from(validated.checks)
            .map_err(|_| SubgroupLatticeErrorV1::ArithmeticOverflow)?;
        for generator in &subgroup.generators {
            if *generator >= group.multiplication.len() {
                return Err(SubgroupLatticeErrorV1::ReceiptMismatch);
            }
            mask = close_generated_subgroup(
                group,
                &validated.inverses,
                mask,
                *generator,
                &mut work_units,
                receipt.budget.maximum_work_units,
            )?;
        }
        if subgroup_mask_to_elements(mask, group.multiplication.len()) != subgroup.elements {
            return Err(SubgroupLatticeErrorV1::ReceiptMismatch);
        }
    }
    Ok(())
}

fn validate_problem(
    problem: &SearchProblemV1,
    budget: QualificationBudgetV1,
) -> Result<(), QuotientErrorV1> {
    if !nonempty(&problem.problem_id) {
        return Err(QuotientErrorV1::InvalidIdentifier);
    }
    let count = problem.candidate_ids.len();
    if count == 0 {
        return Err(QuotientErrorV1::EmptyCandidateUniverse);
    }
    if count > budget.maximum_candidates {
        return Err(QuotientErrorV1::CandidateCountExceeded);
    }
    if problem.semantics.len() != count {
        return Err(QuotientErrorV1::CandidateSemanticsLengthMismatch);
    }
    let mut unique = BTreeSet::new();
    for candidate_id in &problem.candidate_ids {
        if !nonempty(candidate_id) {
            return Err(QuotientErrorV1::InvalidIdentifier);
        }
        if !unique.insert(candidate_id) {
            return Err(QuotientErrorV1::DuplicateCandidateId);
        }
    }
    for semantics in &problem.semantics {
        if !nonempty(&semantics.boundary_condition_id)
            || !nonempty(&semantics.output_equivalence_id)
        {
            return Err(QuotientErrorV1::InvalidIdentifier);
        }
    }
    Ok(())
}

fn validate_action(
    action: &FiniteActionV1,
    candidate_count: usize,
    budget: QualificationBudgetV1,
) -> Result<ValidatedActionV1, QuotientErrorV1> {
    if !nonempty(&action.action_id) {
        return Err(QuotientErrorV1::InvalidIdentifier);
    }
    let group = validate_group(&action.group, budget)?;
    let order = action.group.multiplication.len();
    if action.permutations.len() != order {
        return Err(QuotientErrorV1::ActionCountMismatch);
    }
    for permutation in &action.permutations {
        if permutation.len() != candidate_count {
            return Err(QuotientErrorV1::PermutationLengthMismatch);
        }
        let mut seen = vec![false; candidate_count];
        for image in permutation {
            if *image >= candidate_count || seen[*image] {
                return Err(QuotientErrorV1::NotPermutation);
            }
            seen[*image] = true;
        }
    }
    for candidate in 0..candidate_count {
        if action.permutations[action.group.identity][candidate] != candidate {
            return Err(QuotientErrorV1::IdentityActionMismatch);
        }
    }
    let mut checks = candidate_count;
    for left in 0..order {
        for right in 0..order {
            let product = action.group.multiplication[left][right];
            for candidate in 0..candidate_count {
                checks += 1;
                if action.permutations[left][action.permutations[right][candidate]]
                    != action.permutations[product][candidate]
                {
                    return Err(QuotientErrorV1::ActionLawViolation);
                }
            }
        }
    }
    Ok(ValidatedActionV1 { group, checks })
}

fn check_task_invariance(
    problem: &SearchProblemV1,
    action: &FiniteActionV1,
) -> Result<usize, TaskInvarianceViolationV1> {
    let mut checks = 0usize;
    for (element, permutation) in action.permutations.iter().enumerate() {
        for (source, image) in permutation.iter().copied().enumerate() {
            checks += 1;
            if problem.semantics[source] != problem.semantics[image] {
                return Err(TaskInvarianceViolationV1 {
                    group_element: element,
                    source_index: source,
                    image_index: image,
                    source_candidate_id: problem.candidate_ids[source].clone(),
                    image_candidate_id: problem.candidate_ids[image].clone(),
                });
            }
        }
    }
    Ok(checks)
}

fn build_orbits(
    problem: &SearchProblemV1,
    action: &FiniteActionV1,
    inverses: &[usize],
) -> Vec<OrbitV1> {
    let count = problem.candidate_ids.len();
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by(|a, b| problem.candidate_ids[*a].cmp(&problem.candidate_ids[*b]));
    let mut visited = vec![false; count];
    let mut orbits = Vec::new();

    for representative in order {
        if visited[representative] {
            continue;
        }
        let mut indices = BTreeSet::new();
        for permutation in &action.permutations {
            indices.insert(permutation[representative]);
        }
        for index in &indices {
            visited[*index] = true;
        }
        let mut members = Vec::with_capacity(indices.len());
        for index in indices {
            let element = action
                .permutations
                .iter()
                .position(|permutation| permutation[representative] == index)
                .expect("every orbit member has a witness");
            members.push(MemberWitnessV1 {
                candidate_index: index,
                candidate_id: problem.candidate_ids[index].clone(),
                group_element: element,
                inverse_group_element: inverses[element],
            });
        }
        members.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));
        let stabilizer_group_elements = action
            .permutations
            .iter()
            .enumerate()
            .filter_map(|(element, permutation)| {
                (permutation[representative] == representative).then_some(element)
            })
            .collect();

        orbits.push(OrbitV1 {
            representative_index: representative,
            representative_id: problem.candidate_ids[representative].clone(),
            members,
            stabilizer_group_elements,
        });
    }
    orbits.sort_by(|a, b| a.representative_id.cmp(&b.representative_id));
    orbits
}

fn stabilizer_evidence_counts(
    group_order: usize,
    orbits: &[OrbitV1],
) -> Result<(usize, usize, usize, usize), QuotientErrorV1> {
    let orbit_count = orbits.len();
    let membership_checks = group_order
        .checked_mul(orbit_count)
        .ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    let mut closure_checks = 0usize;
    let mut stabilizer_witnesses = 0usize;
    for orbit in orbits {
        let stabilizer_order = orbit.stabilizer_group_elements.len();
        let local_closure_checks = stabilizer_order
            .checked_mul(stabilizer_order)
            .and_then(|value| value.checked_add(stabilizer_order))
            .ok_or(QuotientErrorV1::ArithmeticOverflow)?;
        closure_checks = closure_checks
            .checked_add(local_closure_checks)
            .ok_or(QuotientErrorV1::ArithmeticOverflow)?;
        stabilizer_witnesses = stabilizer_witnesses
            .checked_add(stabilizer_order)
            .ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    }
    Ok((
        membership_checks,
        closure_checks,
        stabilizer_witnesses,
        orbit_count,
    ))
}

pub fn compile_quotient(
    problem: &SearchProblemV1,
    action: &FiniteActionV1,
    budget: QualificationBudgetV1,
) -> Result<CompileOutcomeV1, QuotientErrorV1> {
    let group_order = action.group.multiplication.len();
    let candidate_count = problem.candidate_ids.len();
    preflight(group_order, candidate_count, budget)?;
    validate_problem(problem, budget)?;
    let validated = validate_action(action, candidate_count, budget)?;
    let task_checks = match check_task_invariance(problem, action) {
        Ok(checks) => checks,
        Err(violation) => return Ok(CompileOutcomeV1::Blocked(violation)),
    };
    let orbits = build_orbits(problem, action, &validated.group.inverses);
    let (
        stabilizer_membership_checks,
        stabilizer_subgroup_closure_checks,
        stabilizer_witnesses,
        orbit_stabilizer_checks,
    ) = stabilizer_evidence_counts(group_order, &orbits)?;
    Ok(CompileOutcomeV1::Authorized(QuotientCertificateV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        action_id: action.action_id.clone(),
        group_id: action.group.group_id.clone(),
        group_order,
        candidate_count,
        pruning_authority: PruningAuthorityV1::ExactFiniteGroupAndExactTaskSemanticPreservation,
        orbits,
        evidence: CertificateEvidenceV1 {
            group_axiom_checks: validated.group.checks,
            action_law_checks: validated.checks,
            task_semantic_checks: task_checks,
            orbit_member_witnesses: candidate_count,
            stabilizer_membership_checks,
            stabilizer_subgroup_closure_checks,
            stabilizer_witnesses,
            orbit_stabilizer_checks,
            complete_domain_cover: true,
        },
    }))
}

/// Replays group, action, task, orbit, witness, ordering, and coverage checks.
pub fn verify_certificate(
    problem: &SearchProblemV1,
    action: &FiniteActionV1,
    certificate: &QuotientCertificateV1,
    budget: QualificationBudgetV1,
) -> Result<VerificationReceiptV1, QuotientErrorV1> {
    let group_order = action.group.multiplication.len();
    let candidate_count = problem.candidate_ids.len();
    preflight(group_order, candidate_count, budget)?;
    validate_problem(problem, budget)?;
    let validated = validate_action(action, candidate_count, budget)?;

    if certificate.schema_version != 1
        || certificate.problem_id != problem.problem_id
        || certificate.action_id != action.action_id
        || certificate.group_id != action.group.group_id
        || certificate.group_order != group_order
        || certificate.candidate_count != candidate_count
    {
        return Err(QuotientErrorV1::CertificateIdentityMismatch);
    }
    if certificate.pruning_authority
        != PruningAuthorityV1::ExactFiniteGroupAndExactTaskSemanticPreservation
    {
        return Err(QuotientErrorV1::WrongPruningAuthority);
    }
    let task_checks = check_task_invariance(problem, action)
        .map_err(|_| QuotientErrorV1::TaskSemanticsMismatch)?;

    let mut covered = vec![false; candidate_count];
    let mut witnessed = 0usize;
    let mut prior_rep: Option<&str> = None;
    for orbit in &certificate.orbits {
        if orbit.members.is_empty() {
            return Err(QuotientErrorV1::EmptyOrbit);
        }
        let rep = orbit.representative_index;
        if rep >= candidate_count {
            return Err(QuotientErrorV1::CandidateIndexOutOfRange);
        }
        if problem.candidate_ids[rep] != orbit.representative_id {
            return Err(QuotientErrorV1::CandidateIdMismatch);
        }
        if prior_rep.is_some_and(|previous| previous >= orbit.representative_id.as_str()) {
            return Err(QuotientErrorV1::NonCanonicalOrdering);
        }
        prior_rep = Some(&orbit.representative_id);

        let mut actual_members = BTreeSet::new();
        let mut prior_member: Option<&str> = None;
        for member in &orbit.members {
            let index = member.candidate_index;
            if index >= candidate_count {
                return Err(QuotientErrorV1::CandidateIndexOutOfRange);
            }
            if problem.candidate_ids[index] != member.candidate_id {
                return Err(QuotientErrorV1::CandidateIdMismatch);
            }
            if prior_member.is_some_and(|previous| previous >= member.candidate_id.as_str()) {
                return Err(QuotientErrorV1::NonCanonicalOrdering);
            }
            prior_member = Some(&member.candidate_id);
            if covered[index] {
                return Err(QuotientErrorV1::OverlappingOrbit);
            }
            covered[index] = true;
            actual_members.insert(index);
            witnessed += 1;

            if member.group_element >= group_order
                || action.permutations[member.group_element][rep] != index
            {
                return Err(QuotientErrorV1::BadWitness);
            }
            if member.inverse_group_element != validated.group.inverses[member.group_element]
                || action.permutations[member.inverse_group_element][index] != rep
            {
                return Err(QuotientErrorV1::BadInverseWitness);
            }
            let first_witness = action
                .permutations
                .iter()
                .position(|permutation| permutation[rep] == index)
                .ok_or(QuotientErrorV1::BadWitness)?;
            if member.group_element != first_witness {
                return Err(QuotientErrorV1::BadWitness);
            }
            if problem.semantics[rep] != problem.semantics[index] {
                return Err(QuotientErrorV1::TaskSemanticsMismatch);
            }
        }

        let minimum_id = actual_members
            .iter()
            .map(|index| problem.candidate_ids[*index].as_str())
            .min()
            .ok_or(QuotientErrorV1::EmptyOrbit)?;
        if minimum_id != orbit.representative_id {
            return Err(QuotientErrorV1::BadRepresentative);
        }
        let expected_members: BTreeSet<usize> = action
            .permutations
            .iter()
            .map(|permutation| permutation[rep])
            .collect();
        if expected_members != actual_members {
            return Err(QuotientErrorV1::IncompleteOrbit);
        }

        let expected_stabilizer: Vec<usize> = action
            .permutations
            .iter()
            .enumerate()
            .filter_map(|(element, permutation)| (permutation[rep] == rep).then_some(element))
            .collect();
        if orbit.stabilizer_group_elements != expected_stabilizer {
            return Err(QuotientErrorV1::BadStabilizerWitness);
        }
        let stabilizer_set: BTreeSet<usize> =
            orbit.stabilizer_group_elements.iter().copied().collect();
        for element in &orbit.stabilizer_group_elements {
            if !stabilizer_set.contains(&validated.group.inverses[*element]) {
                return Err(QuotientErrorV1::StabilizerNotSubgroup);
            }
            for other in &orbit.stabilizer_group_elements {
                let product = action.group.multiplication[*element][*other];
                if !stabilizer_set.contains(&product) {
                    return Err(QuotientErrorV1::StabilizerNotSubgroup);
                }
            }
        }

        let orbit_stabilizer_product = orbit
            .members
            .len()
            .checked_mul(orbit.stabilizer_group_elements.len())
            .ok_or(QuotientErrorV1::ArithmeticOverflow)?;
        if orbit_stabilizer_product != group_order {
            return Err(QuotientErrorV1::OrbitStabilizerMismatch);
        }
    }
    if covered.iter().any(|value| !value) || witnessed != candidate_count {
        return Err(QuotientErrorV1::IncompleteDomainCover);
    }

    let (
        stabilizer_membership_checks,
        stabilizer_subgroup_closure_checks,
        stabilizer_witnesses,
        orbit_stabilizer_checks,
    ) = stabilizer_evidence_counts(group_order, &certificate.orbits)?;
    let expected_evidence = CertificateEvidenceV1 {
        group_axiom_checks: validated.group.checks,
        action_law_checks: validated.checks,
        task_semantic_checks: task_checks,
        orbit_member_witnesses: candidate_count,
        stabilizer_membership_checks,
        stabilizer_subgroup_closure_checks,
        stabilizer_witnesses,
        orbit_stabilizer_checks,
        complete_domain_cover: true,
    };
    if certificate.evidence != expected_evidence {
        return Err(QuotientErrorV1::EvidenceCountMismatch);
    }

    Ok(VerificationReceiptV1 {
        problem_id: problem.problem_id.clone(),
        action_id: action.action_id.clone(),
        group_id: action.group.group_id.clone(),
        group_order,
        candidate_count,
        orbit_count: certificate.orbits.len(),
        witnessed_member_count: witnessed,
        group_axiom_checks: expected_evidence.group_axiom_checks,
        action_law_checks: expected_evidence.action_law_checks,
        task_semantic_checks: expected_evidence.task_semantic_checks,
    })
}

pub fn lift_representative_solution(
    problem: &SearchProblemV1,
    action: &FiniteActionV1,
    certificate: &QuotientCertificateV1,
    representative_index: usize,
    budget: QualificationBudgetV1,
) -> Result<Vec<LiftedCandidateV1>, QuotientErrorV1> {
    verify_certificate(problem, action, certificate, budget)?;
    let orbit = certificate
        .orbits
        .iter()
        .find(|orbit| orbit.representative_index == representative_index)
        .ok_or(QuotientErrorV1::RequestedCandidateIsNotRepresentative)?;
    Ok(orbit
        .members
        .iter()
        .map(|member| LiftedCandidateV1 {
            candidate_index: member.candidate_index,
            candidate_id: member.candidate_id.clone(),
            group_element_from_representative: member.group_element,
            inverse_group_element: member.inverse_group_element,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c2_action() -> FiniteActionV1 {
        FiniteActionV1 {
            action_id: "pair-swap-action-v1".to_owned(),
            group: FiniteGroupV1 {
                group_id: "C2-v1".to_owned(),
                identity: 0,
                multiplication: vec![vec![0, 1], vec![1, 0]],
            },
            permutations: vec![vec![0, 1, 2, 3], vec![1, 0, 3, 2]],
        }
    }

    fn d4_action() -> FiniteActionV1 {
        let mut multiplication = vec![vec![0; 8]; 8];
        for left in 0..8 {
            let left_rotation = left % 4;
            let left_reflection = left / 4;
            for right in 0..8 {
                let right_rotation = right % 4;
                let right_reflection = right / 4;
                let signed_rotation = if left_reflection == 0 {
                    right_rotation
                } else {
                    (4 - right_rotation) % 4
                };
                let product_rotation = (left_rotation + signed_rotation) % 4;
                let product_reflection = left_reflection ^ right_reflection;
                multiplication[left][right] = product_reflection * 4 + product_rotation;
            }
        }

        let mut permutations = Vec::with_capacity(8);
        for element in 0..8 {
            let rotation = element % 4;
            let reflection = element / 4;
            let mut permutation = vec![0; 4];
            for vertex in 0..4 {
                let signed_vertex = if reflection == 0 {
                    vertex
                } else {
                    (4 - vertex) % 4
                };
                permutation[vertex] = (rotation + signed_vertex) % 4;
            }
            permutations.push(permutation);
        }

        let mut pair_permutations = Vec::with_capacity(8);
        for permutation in &permutations {
            let mut action = vec![0; 16];
            for source in 0..4 {
                for target in 0..4 {
                    action[source * 4 + target] = permutation[source] * 4 + permutation[target];
                }
            }
            pair_permutations.push(action);
        }

        FiniteActionV1 {
            action_id: "D4-ordered-vertex-pair-action-v1".to_owned(),
            group: FiniteGroupV1 {
                group_id: "D4-order-8-v1".to_owned(),
                identity: 0,
                multiplication,
            },
            permutations: pair_permutations,
        }
    }

    fn semantics(cost: i64, constraint: i64, boundary: &str, output: &str) -> CandidateSemanticsV1 {
        CandidateSemanticsV1 {
            in_search_domain: true,
            admissible: true,
            feasible: true,
            objective_cost: cost,
            constraint_signature: vec![constraint],
            boundary_condition_id: boundary.to_owned(),
            output_equivalence_id: output.to_owned(),
        }
    }

    fn c2_problem() -> SearchProblemV1 {
        SearchProblemV1 {
            problem_id: "two-pair-search-v1".to_owned(),
            candidate_ids: vec![
                "A0".to_owned(),
                "A1".to_owned(),
                "B0".to_owned(),
                "B1".to_owned(),
            ],
            semantics: vec![
                semantics(10, 2, "boundary-a", "output-a"),
                semantics(10, 2, "boundary-a", "output-a"),
                semantics(20, 3, "boundary-b", "output-b"),
                semantics(20, 3, "boundary-b", "output-b"),
            ],
        }
    }

    fn exhaustive_oracle(action: &FiniteActionV1, count: usize) -> Vec<BTreeSet<usize>> {
        let mut remaining: BTreeSet<usize> = (0..count).collect();
        let mut result = Vec::new();
        while let Some(seed) = remaining.iter().next().copied() {
            let orbit: BTreeSet<usize> = action
                .permutations
                .iter()
                .map(|permutation| permutation[seed])
                .collect();
            for member in &orbit {
                remaining.remove(member);
            }
            result.push(orbit);
        }
        result
    }

    fn authorized_fixture() -> (SearchProblemV1, FiniteActionV1, QuotientCertificateV1) {
        let problem = c2_problem();
        let action = c2_action();
        let outcome =
            compile_quotient(&problem, &action, QualificationBudgetV1::default()).unwrap();
        let CompileOutcomeV1::Authorized(certificate) = outcome else {
            panic!("expected an exact task-preserving quotient");
        };
        (problem, action, certificate)
    }


    #[test]
    fn d4_subgroup_lattice_matches_exact_known_structure() {
        let group = d4_action().group;
        let receipt =
            enumerate_subgroup_lattice(&group, SubgroupLatticeBudgetV1::default()).unwrap();
        assert_eq!(receipt.subgroups.len(), 10);
        let mut orders: Vec<usize> = receipt
            .subgroups
            .iter()
            .map(|subgroup| subgroup.elements.len())
            .collect();
        orders.sort_unstable();
        assert_eq!(orders, vec![1, 2, 2, 2, 2, 2, 4, 4, 4, 8]);
        assert_eq!(receipt.maximal_proper_subgroup_indices.len(), 3);
        assert!(receipt.maximal_proper_subgroup_indices.iter().all(|index| {
            receipt.subgroups[*index].elements.len() == 4
        }));
        for subgroup in &receipt.subgroups {
            assert!(!subgroup.generators.iter().any(|generator| {
                !subgroup.elements.contains(generator)
            }));
        }
        verify_subgroup_lattice_receipt(&group, &receipt).unwrap();
    }

    #[test]
    fn cyclic_c4_lattice_has_one_unique_maximal_proper_subgroup() {
        let group = FiniteGroupV1 {
            group_id: "C4-subgroup-lattice-fixture".to_owned(),
            identity: 0,
            multiplication: vec![
                vec![0, 1, 2, 3],
                vec![1, 2, 3, 0],
                vec![2, 3, 0, 1],
                vec![3, 0, 1, 2],
            ],
        };
        let receipt =
            enumerate_subgroup_lattice(&group, SubgroupLatticeBudgetV1::default()).unwrap();
        assert_eq!(receipt.subgroups.len(), 3);
        assert_eq!(receipt.maximal_proper_subgroup_indices.len(), 1);
        let maximal = &receipt.subgroups[receipt.maximal_proper_subgroup_indices[0]];
        assert_eq!(maximal.elements, vec![0, 2]);
        verify_subgroup_lattice_receipt(&group, &receipt).unwrap();
    }

    #[test]
    fn subgroup_lattice_work_and_subgroup_count_budgets_fail_closed() {
        let group = d4_action().group;
        assert_eq!(
            enumerate_subgroup_lattice(
                &group,
                SubgroupLatticeBudgetV1 {
                    maximum_group_order: 8,
                    maximum_subgroups: 2,
                    maximum_work_units: 1_000_000,
                }
            ),
            Err(SubgroupLatticeErrorV1::SubgroupCountExceeded)
        );
        assert_eq!(
            enumerate_subgroup_lattice(
                &group,
                SubgroupLatticeBudgetV1 {
                    maximum_group_order: 8,
                    maximum_subgroups: 100,
                    maximum_work_units: 1,
                }
            ),
            Err(SubgroupLatticeErrorV1::WorkBudgetExceeded)
        );
    }

    #[test]
    fn subgroup_lattice_receipt_rejects_forgery_and_group_mutation() {
        let group = d4_action().group;
        let mut receipt =
            enumerate_subgroup_lattice(&group, SubgroupLatticeBudgetV1::default()).unwrap();
        verify_subgroup_lattice_receipt(&group, &receipt).unwrap();
        receipt.maximal_proper_subgroup_indices.pop();
        assert_eq!(
            verify_subgroup_lattice_receipt(&group, &receipt),
            Err(SubgroupLatticeErrorV1::ReceiptMismatch)
        );

        let original =
            enumerate_subgroup_lattice(&group, SubgroupLatticeBudgetV1::default()).unwrap();
        let mut changed = group.clone();
        changed.group_id = "same-table-different-group-identity".to_owned();
        assert_eq!(
            verify_subgroup_lattice_receipt(&changed, &original),
            Err(SubgroupLatticeErrorV1::ReceiptMismatch)
        );
    }

    #[test]
    fn subgroup_lattice_rejects_non_groups_using_the_shared_group_validator() {
        let group = FiniteGroupV1 {
            group_id: "not-a-group".to_owned(),
            identity: 0,
            multiplication: vec![vec![0, 1], vec![1, 1]],
        };
        assert_eq!(
            enumerate_subgroup_lattice(&group, SubgroupLatticeBudgetV1::default()),
            Err(SubgroupLatticeErrorV1::InvalidGroup(
                QuotientErrorV1::MultiplicationTableNotLatin
            ))
        );
    }

    #[test]
    fn c2_quotient_matches_exhaustive_oracle_and_lifts_all_members() {
        let (problem, action, certificate) = authorized_fixture();
        let receipt = verify_certificate(
            &problem,
            &action,
            &certificate,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        let compiled: Vec<BTreeSet<usize>> = certificate
            .orbits
            .iter()
            .map(|orbit| {
                orbit
                    .members
                    .iter()
                    .map(|member| member.candidate_index)
                    .collect()
            })
            .collect();
        assert_eq!(
            compiled,
            exhaustive_oracle(&action, problem.candidate_ids.len())
        );
        assert_eq!(receipt.orbit_count, 2);
        assert_eq!(receipt.witnessed_member_count, 4);
        let lifted = lift_representative_solution(
            &problem,
            &action,
            &certificate,
            0,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(
            lifted
                .iter()
                .map(|member| member.candidate_id.as_str())
                .collect::<Vec<_>>(),
            vec!["A0", "A1"]
        );
        assert_eq!(
            lift_representative_solution(
                &problem,
                &action,
                &certificate,
                1,
                QualificationBudgetV1::default()
            ),
            Err(QuotientErrorV1::RequestedCandidateIsNotRepresentative)
        );
    }

    #[test]
    fn boundary_condition_change_blocks_pruning() {
        let action = c2_action();
        let mut problem = c2_problem();
        problem.semantics[1].boundary_condition_id = "different-boundary".to_owned();
        let outcome =
            compile_quotient(&problem, &action, QualificationBudgetV1::default()).unwrap();
        assert_eq!(
            outcome,
            CompileOutcomeV1::Blocked(TaskInvarianceViolationV1 {
                group_element: 1,
                source_index: 0,
                image_index: 1,
                source_candidate_id: "A0".to_owned(),
                image_candidate_id: "A1".to_owned(),
            })
        );
    }

    #[test]
    fn latin_loop_without_associativity_is_rejected_as_a_group() {
        let group = FiniteGroupV1 {
            group_id: "nonassociative-latin-loop-order-six".to_owned(),
            identity: 0,
            multiplication: vec![
                vec![0, 1, 2, 3, 4, 5],
                vec![1, 3, 5, 2, 0, 4],
                vec![2, 5, 4, 1, 3, 0],
                vec![3, 2, 0, 4, 5, 1],
                vec![4, 0, 3, 5, 1, 2],
                vec![5, 4, 1, 0, 2, 3],
            ],
        };
        assert_eq!(
            validate_group(&group, QualificationBudgetV1::default()),
            Err(QuotientErrorV1::NonAssociativeGroup)
        );
    }

    #[test]
    fn non_latin_multiplication_table_is_rejected() {
        let mut action = c2_action();
        action.group.multiplication = vec![vec![0, 1], vec![1, 1]];
        assert_eq!(
            validate_action(&action, 4, QualificationBudgetV1::default()),
            Err(QuotientErrorV1::MultiplicationTableNotLatin)
        );
    }

    #[test]
    fn non_bijective_action_is_rejected() {
        let mut action = c2_action();
        action.permutations[1] = vec![0, 0, 2, 3];
        assert_eq!(
            validate_action(&action, 4, QualificationBudgetV1::default()),
            Err(QuotientErrorV1::NotPermutation)
        );
    }

    #[test]
    fn omitted_orbit_member_is_rejected() {
        let (problem, action, mut certificate) = authorized_fixture();
        certificate.orbits[0].members.pop();
        assert_eq!(
            verify_certificate(
                &problem,
                &action,
                &certificate,
                QualificationBudgetV1::default()
            ),
            Err(QuotientErrorV1::IncompleteOrbit)
        );
    }

    #[test]
    fn incorrect_inverse_witness_is_rejected() {
        let (problem, action, mut certificate) = authorized_fixture();
        certificate.orbits[0].members[1].inverse_group_element = 0;
        assert_eq!(
            verify_certificate(
                &problem,
                &action,
                &certificate,
                QualificationBudgetV1::default()
            ),
            Err(QuotientErrorV1::BadInverseWitness)
        );
    }

    #[test]
    fn trivial_group_produces_singleton_orbits() {
        let problem = SearchProblemV1 {
            problem_id: "trivial-action-search".to_owned(),
            candidate_ids: vec!["c".to_owned(), "a".to_owned(), "b".to_owned()],
            semantics: vec![
                semantics(3, 0, "bc", "oc"),
                semantics(1, 0, "ba", "oa"),
                semantics(2, 0, "bb", "ob"),
            ],
        };
        let action = FiniteActionV1 {
            action_id: "trivial-action".to_owned(),
            group: FiniteGroupV1 {
                group_id: "C1".to_owned(),
                identity: 0,
                multiplication: vec![vec![0]],
            },
            permutations: vec![vec![0, 1, 2]],
        };
        let outcome =
            compile_quotient(&problem, &action, QualificationBudgetV1::default()).unwrap();
        let CompileOutcomeV1::Authorized(certificate) = outcome else {
            panic!("expected trivial quotient");
        };
        assert_eq!(
            certificate
                .orbits
                .iter()
                .map(|orbit| orbit.representative_id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
        verify_certificate(
            &problem,
            &action,
            &certificate,
            QualificationBudgetV1::default(),
        )
        .unwrap();
    }

    #[test]
    fn cyclic_group_of_order_three_produces_one_exact_orbit() {
        let problem = SearchProblemV1 {
            problem_id: "cyclic-three-search".to_owned(),
            candidate_ids: vec!["x0".to_owned(), "x1".to_owned(), "x2".to_owned()],
            semantics: vec![
                semantics(5, 1, "same-boundary", "same-output"),
                semantics(5, 1, "same-boundary", "same-output"),
                semantics(5, 1, "same-boundary", "same-output"),
            ],
        };
        let action = FiniteActionV1 {
            action_id: "cyclic-three-rotation".to_owned(),
            group: FiniteGroupV1 {
                group_id: "C3".to_owned(),
                identity: 0,
                multiplication: vec![vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1]],
            },
            permutations: vec![vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1]],
        };
        let outcome =
            compile_quotient(&problem, &action, QualificationBudgetV1::default()).unwrap();
        let CompileOutcomeV1::Authorized(certificate) = outcome else {
            panic!("expected an exact C3 quotient");
        };
        assert_eq!(certificate.orbits.len(), 1);
        assert_eq!(certificate.orbits[0].members.len(), 3);
        verify_certificate(
            &problem,
            &action,
            &certificate,
            QualificationBudgetV1::default(),
        )
        .unwrap();
    }

    #[test]
    fn non_abelian_d4_pair_action_matches_three_exhaustive_orbits() {
        let action = d4_action();
        assert_ne!(
            action.group.multiplication[1][4],
            action.group.multiplication[4][1]
        );

        let mut candidate_ids = Vec::with_capacity(16);
        let mut semantics = Vec::with_capacity(16);
        for source in 0usize..4 {
            for target in 0usize..4 {
                let difference = source.abs_diff(target);
                let distance = difference.min(4 - difference);
                candidate_ids.push(format!("p{source}{target}"));
                semantics.push(CandidateSemanticsV1 {
                    in_search_domain: true,
                    admissible: true,
                    feasible: true,
                    objective_cost: distance as i64,
                    constraint_signature: vec![distance as i64],
                    boundary_condition_id: "fixed-square-boundary-v1".to_owned(),
                    output_equivalence_id: format!("distance-class-{distance}"),
                });
            }
        }
        let problem = SearchProblemV1 {
            problem_id: "D4-square-ordered-pairs-v1".to_owned(),
            candidate_ids,
            semantics,
        };

        let outcome =
            compile_quotient(&problem, &action, QualificationBudgetV1::default()).unwrap();
        let CompileOutcomeV1::Authorized(certificate) = outcome else {
            panic!("D4 action must preserve pair-distance task semantics");
        };
        let compiled: Vec<BTreeSet<usize>> = certificate
            .orbits
            .iter()
            .map(|orbit| {
                orbit
                    .members
                    .iter()
                    .map(|member| member.candidate_index)
                    .collect()
            })
            .collect();
        assert_eq!(compiled, exhaustive_oracle(&action, 16));
        assert_eq!(
            certificate
                .orbits
                .iter()
                .map(|orbit| orbit.members.len())
                .collect::<Vec<_>>(),
            vec![4, 8, 4]
        );
        assert_eq!(
            certificate
                .orbits
                .iter()
                .map(|orbit| orbit.stabilizer_group_elements.len())
                .collect::<Vec<_>>(),
            vec![2, 1, 2]
        );
        assert!(
            certificate
                .orbits
                .iter()
                .all(|orbit| { orbit.members.len() * orbit.stabilizer_group_elements.len() == 8 })
        );
        verify_certificate(
            &problem,
            &action,
            &certificate,
            QualificationBudgetV1::default(),
        )
        .unwrap();

        // Compare an actual objective search, not only orbit partitions.
        let exhaustive_best = problem
            .semantics
            .iter()
            .map(|semantics| semantics.objective_cost)
            .min()
            .unwrap();
        let exhaustive_solutions: BTreeSet<usize> = problem
            .semantics
            .iter()
            .enumerate()
            .filter_map(|(index, semantics)| {
                (semantics.objective_cost == exhaustive_best).then_some(index)
            })
            .collect();

        let quotient_best = certificate
            .orbits
            .iter()
            .map(|orbit| problem.semantics[orbit.representative_index].objective_cost)
            .min()
            .unwrap();
        let mut lifted_quotient_solutions = BTreeSet::new();
        for orbit in &certificate.orbits {
            if problem.semantics[orbit.representative_index].objective_cost == quotient_best {
                let lifted = lift_representative_solution(
                    &problem,
                    &action,
                    &certificate,
                    orbit.representative_index,
                    QualificationBudgetV1::default(),
                )
                .unwrap();
                lifted_quotient_solutions.extend(
                    lifted
                        .into_iter()
                        .map(|candidate| candidate.candidate_index),
                );
            }
        }
        assert_eq!(quotient_best, exhaustive_best);
        assert_eq!(lifted_quotient_solutions, exhaustive_solutions);
        assert_eq!(exhaustive_solutions.len(), 4);
        assert_eq!(problem.candidate_ids.len(), 16);
        assert_eq!(certificate.orbits.len(), 3);
        // The exact finite fixture needs 3 representative objective calls
        // instead of 16 exhaustive calls: 16/3 = 5.33x fewer evaluations.
        // This is candidate-work reduction, not measured wall-clock speedup.
        assert_eq!(problem.candidate_ids.len(), 16);
        assert_eq!(certificate.orbits.len(), 3);
    }

    #[test]
    fn incomplete_stabilizer_witness_is_rejected() {
        let (problem, action, mut certificate) = authorized_fixture();
        certificate.orbits[0].stabilizer_group_elements.pop();
        assert_eq!(
            verify_certificate(
                &problem,
                &action,
                &certificate,
                QualificationBudgetV1::default()
            ),
            Err(QuotientErrorV1::BadStabilizerWitness)
        );
    }

    #[test]
    fn work_budget_fails_before_qualification() {
        let budget = QualificationBudgetV1 {
            maximum_group_order: 4,
            maximum_candidates: 4,
            maximum_work_units: 1,
        };
        assert_eq!(
            compile_quotient(&c2_problem(), &c2_action(), budget),
            Err(QuotientErrorV1::WorkBudgetExceeded)
        );
    }

    #[test]
    fn action_law_violation_is_rejected() {
        let mut action = c2_action();
        action.permutations[1] = vec![1, 2, 3, 0];
        assert_eq!(
            validate_action(&action, 4, QualificationBudgetV1::default()),
            Err(QuotientErrorV1::ActionLawViolation)
        );
    }

    #[test]
    fn certificate_cannot_be_replayed_against_changed_task_semantics() {
        let (_problem, action, certificate) = authorized_fixture();
        let mut changed_problem = c2_problem();
        changed_problem.semantics[2].objective_cost += 1;
        assert_eq!(
            verify_certificate(
                &changed_problem,
                &action,
                &certificate,
                QualificationBudgetV1::default()
            ),
            Err(QuotientErrorV1::TaskSemanticsMismatch)
        );
    }
}
