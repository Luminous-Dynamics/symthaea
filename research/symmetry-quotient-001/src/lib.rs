// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact finite-group quotient construction for bounded discrete search tasks.
//! Discovery is not proof; task preservation is a separate fail-closed obligation.

// Explicit indices keep finite multiplication/action tables auditable.
#![allow(clippy::needless_range_loop)]

use std::collections::BTreeSet;
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


/// One independently declared operating regime, boundary-condition family,
/// intervention setting, or task variant over the same candidate universe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskEnvironmentV1 {
    pub environment_id: String,
    pub semantics: Vec<CandidateSemanticsV1>,
}

/// The symmetry that is safe for a family of environments is the intersection
/// of their individual task-preserving stabilizers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiEnvironmentProblemV1 {
    pub problem_id: String,
    pub candidate_ids: Vec<String>,
    pub environments: Vec<TaskEnvironmentV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskFieldV1 {
    InSearchDomain,
    Admissible,
    Feasible,
    ObjectiveCost,
    ConstraintSignature,
    BoundaryConditionId,
    OutputEquivalenceId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvironmentExclusionWitnessV1 {
    pub ambient_group_element: usize,
    pub environment_id: String,
    pub source_index: usize,
    pub image_index: usize,
    pub source_candidate_id: String,
    pub image_candidate_id: String,
    pub differing_fields: Vec<TaskFieldV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommonSubgroupEvidenceV1 {
    pub ambient_group_axiom_checks: usize,
    pub ambient_action_law_checks: usize,
    pub environment_semantic_checks: usize,
    pub subgroup_closure_checks: usize,
    pub retained_ambient_elements: usize,
    pub excluded_ambient_elements: usize,
    pub complete_environment_cover: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommonTaskPreservingSubgroupV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub ambient_action_id: String,
    pub ambient_group_id: String,
    /// Indices refer to the original action's group-element order.
    pub retained_ambient_elements: Vec<usize>,
    pub preserving_environment_ids: Vec<String>,
    /// One concrete counterexample for each excluded ambient element.
    pub exclusion_witnesses: Vec<EnvironmentExclusionWitnessV1>,
    /// The exact restricted action used by every per-environment quotient.
    pub restricted_action: FiniteActionV1,
    pub evidence: CommonSubgroupEvidenceV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvironmentQuotientV1 {
    pub environment_id: String,
    pub certificate: QuotientCertificateV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiEnvironmentQuotientCertificateV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub common_subgroup: CommonTaskPreservingSubgroupV1,
    pub environment_quotients: Vec<EnvironmentQuotientV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiEnvironmentVerificationReceiptV1 {
    pub problem_id: String,
    pub environment_count: usize,
    pub ambient_group_order: usize,
    pub common_subgroup_order: usize,
    pub candidate_count: usize,
    pub per_environment_orbit_counts: Vec<(String, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextualEnvironmentQuotientV1 {
    /// Must equal the exact environment_id used to qualify this subgroup.
    pub context_id: String,
    pub task_preserving_subgroup: CommonTaskPreservingSubgroupV1,
    pub certificate: QuotientCertificateV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextualMultiEnvironmentCertificateV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub ambient_action_id: String,
    /// Canonically ordered exactly as the declared environment list.
    pub environment_quotients: Vec<ContextualEnvironmentQuotientV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextualEnvironmentVerificationV1 {
    pub context_id: String,
    pub preserving_subgroup_order: usize,
    pub orbit_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextualMultiEnvironmentVerificationReceiptV1 {
    pub problem_id: String,
    pub ambient_group_order: usize,
    pub candidate_count: usize,
    pub environments: Vec<ContextualEnvironmentVerificationV1>,
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
    InvalidEnvironmentCount,
    InvalidEnvironmentId,
    DuplicateEnvironmentId,
    EnvironmentSemanticsLengthMismatch,
    CommonTaskPreserversNotSubgroup,
    EnvironmentCertificateMismatch,
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

fn multi_environment_validation_work(
    problem: &MultiEnvironmentProblemV1,
) -> Result<usize, QuotientErrorV1> {
    let candidate_count = problem.candidate_ids.len();
    let per_candidate = ceil_log2(candidate_count)
        .checked_add(1)
        .ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    problem
        .environments
        .len()
        .checked_mul(candidate_count)
        .and_then(|value| value.checked_mul(per_candidate))
        .ok_or(QuotientErrorV1::ArithmeticOverflow)
}

fn validate_multi_environment_problem(
    problem: &MultiEnvironmentProblemV1,
    budget: QualificationBudgetV1,
) -> Result<(), QuotientErrorV1> {
    if !nonempty(&problem.problem_id) {
        return Err(QuotientErrorV1::InvalidIdentifier);
    }
    if problem.environments.is_empty() {
        return Err(QuotientErrorV1::InvalidEnvironmentCount);
    }
    if multi_environment_validation_work(problem)? > budget.maximum_work_units {
        return Err(QuotientErrorV1::WorkBudgetExceeded);
    }
    let mut environment_ids = BTreeSet::new();
    for environment in &problem.environments {
        if !nonempty(&environment.environment_id) {
            return Err(QuotientErrorV1::InvalidEnvironmentId);
        }
        if !environment_ids.insert(environment.environment_id.as_str()) {
            return Err(QuotientErrorV1::DuplicateEnvironmentId);
        }
        let search_problem = SearchProblemV1 {
            problem_id: format!("{}::{}", problem.problem_id, environment.environment_id),
            candidate_ids: problem.candidate_ids.clone(),
            semantics: environment.semantics.clone(),
        };
        if environment.semantics.len() != problem.candidate_ids.len() {
            return Err(QuotientErrorV1::EnvironmentSemanticsLengthMismatch);
        }
        validate_problem(&search_problem, budget)?;
    }
    Ok(())
}

fn differing_task_fields(
    left: &CandidateSemanticsV1,
    right: &CandidateSemanticsV1,
) -> Vec<TaskFieldV1> {
    let mut fields = Vec::new();
    if left.in_search_domain != right.in_search_domain {
        fields.push(TaskFieldV1::InSearchDomain);
    }
    if left.admissible != right.admissible {
        fields.push(TaskFieldV1::Admissible);
    }
    if left.feasible != right.feasible {
        fields.push(TaskFieldV1::Feasible);
    }
    if left.objective_cost != right.objective_cost {
        fields.push(TaskFieldV1::ObjectiveCost);
    }
    if left.constraint_signature != right.constraint_signature {
        fields.push(TaskFieldV1::ConstraintSignature);
    }
    if left.boundary_condition_id != right.boundary_condition_id {
        fields.push(TaskFieldV1::BoundaryConditionId);
    }
    if left.output_equivalence_id != right.output_equivalence_id {
        fields.push(TaskFieldV1::OutputEquivalenceId);
    }
    fields
}

/// Find the exact intersection of task-preserving stabilizers across every
/// declared environment. An ambient element is retained only if it preserves
/// every candidate's complete semantics in every environment. The retained
/// set is then checked as a subgroup and materialized as a restricted action.
pub fn qualify_common_task_preserving_subgroup(
    problem: &MultiEnvironmentProblemV1,
    ambient_action: &FiniteActionV1,
    budget: QualificationBudgetV1,
) -> Result<CommonTaskPreservingSubgroupV1, QuotientErrorV1> {
    let group_order = ambient_action.group.multiplication.len();
    let candidate_count = problem.candidate_ids.len();
    let validation_work = multi_environment_validation_work(problem)?;
    let environment_checks = problem
        .environments
        .len()
        .checked_mul(group_order)
        .and_then(|value| value.checked_mul(candidate_count))
        .ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    let overhead_work = validation_work
        .checked_add(environment_checks)
        .ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    let remaining_work = budget
        .maximum_work_units
        .checked_sub(overhead_work)
        .ok_or(QuotientErrorV1::WorkBudgetExceeded)?;
    let adjusted_budget = QualificationBudgetV1 {
        maximum_work_units: remaining_work,
        ..budget
    };
    preflight(group_order, candidate_count, adjusted_budget)?;
    let validation_budget = QualificationBudgetV1 {
        maximum_work_units: validation_work,
        ..budget
    };
    validate_multi_environment_problem(problem, validation_budget)?;
    let validated = validate_action(ambient_action, candidate_count, adjusted_budget)?;

    let mut retained = Vec::new();
    let mut exclusion_witnesses = Vec::new();
    for (group_element, permutation) in ambient_action.permutations.iter().enumerate() {
        let mut first_violation = None;
        'environment_scan: for environment in &problem.environments {
            for (source_index, image_index) in permutation.iter().copied().enumerate() {
                let differing_fields = differing_task_fields(
                    &environment.semantics[source_index],
                    &environment.semantics[image_index],
                );
                if !differing_fields.is_empty() {
                    first_violation = Some(EnvironmentExclusionWitnessV1 {
                        ambient_group_element: group_element,
                        environment_id: environment.environment_id.clone(),
                        source_index,
                        image_index,
                        source_candidate_id: problem.candidate_ids[source_index].clone(),
                        image_candidate_id: problem.candidate_ids[image_index].clone(),
                        differing_fields,
                    });
                    break 'environment_scan;
                }
            }
        }
        if let Some(witness) = first_violation {
            exclusion_witnesses.push(witness);
        } else {
            retained.push(group_element);
        }
    }

    if !retained.contains(&ambient_action.group.identity) {
        return Err(QuotientErrorV1::TaskSemanticsMismatch);
    }
    let mut ambient_to_restricted = vec![usize::MAX; group_order];
    for (restricted_index, ambient_index) in retained.iter().copied().enumerate() {
        ambient_to_restricted[ambient_index] = restricted_index;
    }

    let mut restricted_multiplication = vec![vec![0; retained.len()]; retained.len()];
    let mut subgroup_closure_checks = 0usize;
    for (left_index, left_ambient) in retained.iter().copied().enumerate() {
        for (right_index, right_ambient) in retained.iter().copied().enumerate() {
            subgroup_closure_checks += 1;
            let product = ambient_action.group.multiplication[left_ambient][right_ambient];
            let restricted_product = ambient_to_restricted[product];
            if restricted_product == usize::MAX {
                return Err(QuotientErrorV1::CommonTaskPreserversNotSubgroup);
            }
            restricted_multiplication[left_index][right_index] = restricted_product;
        }
    }
    let restricted_identity = ambient_to_restricted[ambient_action.group.identity];
    for ambient_element in &retained {
        if ambient_to_restricted[validated.group.inverses[*ambient_element]] == usize::MAX {
            return Err(QuotientErrorV1::CommonTaskPreserversNotSubgroup);
        }
    }

    let restricted_group = FiniteGroupV1 {
        group_id: format!("{}::common-task-preserving", ambient_action.group.group_id),
        identity: restricted_identity,
        multiplication: restricted_multiplication,
    };
    let restricted_permutations: Vec<Vec<usize>> = retained
        .iter()
        .map(|ambient_element| ambient_action.permutations[*ambient_element].clone())
        .collect();
    let restricted_action = FiniteActionV1 {
        action_id: format!("{}::common-task-preserving", ambient_action.action_id),
        group: restricted_group,
        permutations: restricted_permutations,
    };
    validate_action(&restricted_action, candidate_count, budget)?;

    Ok(CommonTaskPreservingSubgroupV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        ambient_action_id: ambient_action.action_id.clone(),
        ambient_group_id: ambient_action.group.group_id.clone(),
        retained_ambient_elements: retained.clone(),
        preserving_environment_ids: problem
            .environments
            .iter()
            .map(|environment| environment.environment_id.clone())
            .collect(),
        exclusion_witnesses,
        restricted_action,
        evidence: CommonSubgroupEvidenceV1 {
            ambient_group_axiom_checks: validated.group.checks,
            ambient_action_law_checks: validated.checks,
            environment_semantic_checks: environment_checks,
            subgroup_closure_checks,
            retained_ambient_elements: retained.len(),
            excluded_ambient_elements: group_order - retained.len(),
            complete_environment_cover: true,
        },
    })
}

/// Compile a separately replayable quotient for every environment, using the
/// exact common subgroup. A symmetry that fails in even one environment is
/// excluded before any quotient receives pruning authority.
fn multi_environment_stage_budget(
    problem: &MultiEnvironmentProblemV1,
    budget: QualificationBudgetV1,
) -> Result<QualificationBudgetV1, QuotientErrorV1> {
    let environment_count = problem.environments.len();
    if environment_count == 0 {
        return Err(QuotientErrorV1::InvalidEnvironmentCount);
    }
    // Reserve a conservative equal share for common-subgroup qualification,
    // every per-environment compile, and every per-environment verification.
    let stage_count = environment_count
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    let maximum_work_units = budget.maximum_work_units / stage_count;
    if maximum_work_units == 0 {
        return Err(QuotientErrorV1::WorkBudgetExceeded);
    }
    Ok(QualificationBudgetV1 {
        maximum_work_units,
        ..budget
    })
}

pub fn compile_multi_environment_quotient(
    problem: &MultiEnvironmentProblemV1,
    ambient_action: &FiniteActionV1,
    budget: QualificationBudgetV1,
) -> Result<MultiEnvironmentQuotientCertificateV1, QuotientErrorV1> {
    let stage_budget = multi_environment_stage_budget(problem, budget)?;
    let common_subgroup =
        qualify_common_task_preserving_subgroup(problem, ambient_action, stage_budget)?;
    let mut environment_quotients = Vec::with_capacity(problem.environments.len());
    for environment in &problem.environments {
        let search_problem = SearchProblemV1 {
            problem_id: format!("{}::{}", problem.problem_id, environment.environment_id),
            candidate_ids: problem.candidate_ids.clone(),
            semantics: environment.semantics.clone(),
        };
        let outcome = compile_quotient(
            &search_problem,
            &common_subgroup.restricted_action,
            stage_budget,
        )?;
        let CompileOutcomeV1::Authorized(certificate) = outcome else {
            return Err(QuotientErrorV1::TaskSemanticsMismatch);
        };
        environment_quotients.push(EnvironmentQuotientV1 {
            environment_id: environment.environment_id.clone(),
            certificate,
        });
    }
    Ok(MultiEnvironmentQuotientCertificateV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        common_subgroup,
        environment_quotients,
    })
}

fn contextual_stage_budget(
    problem: &MultiEnvironmentProblemV1,
    budget: QualificationBudgetV1,
) -> Result<QualificationBudgetV1, QuotientErrorV1> {
    let stage_count = problem
        .environments
        .len()
        .checked_mul(3)
        .and_then(|value| value.checked_add(1))
        .ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    if stage_count == 0 {
        return Err(QuotientErrorV1::InvalidEnvironmentCount);
    }
    // Reserve equal cost envelopes for each local subgroup qualification,
    // quotient compile, and the later independent per-context replay.
    let maximum_work_units = budget.maximum_work_units / stage_count;
    if maximum_work_units == 0 {
        return Err(QuotientErrorV1::WorkBudgetExceeded);
    }
    Ok(QualificationBudgetV1 {
        maximum_work_units,
        ..budget
    })
}

/// Compile a symmetry atlas: each known environment gets its own exact
/// task-preserving subgroup and quotient certificate. No environment borrows
/// another environment's pruning authority; callers must select the proof
/// whose context_id exactly matches the active operating context.
pub fn compile_contextual_multi_environment_quotient(
    problem: &MultiEnvironmentProblemV1,
    ambient_action: &FiniteActionV1,
    budget: QualificationBudgetV1,
) -> Result<ContextualMultiEnvironmentCertificateV1, QuotientErrorV1> {
    let stage_budget = contextual_stage_budget(problem, budget)?;
    validate_multi_environment_problem(problem, stage_budget)?;
    let mut environment_quotients = Vec::with_capacity(problem.environments.len());
    for environment in &problem.environments {
        let local_problem = MultiEnvironmentProblemV1 {
            problem_id: problem.problem_id.clone(),
            candidate_ids: problem.candidate_ids.clone(),
            environments: vec![environment.clone()],
        };
        let subgroup = qualify_common_task_preserving_subgroup(
            &local_problem,
            ambient_action,
            stage_budget,
        )?;
        let search_problem = SearchProblemV1 {
            problem_id: format!("{}::{}", problem.problem_id, environment.environment_id),
            candidate_ids: problem.candidate_ids.clone(),
            semantics: environment.semantics.clone(),
        };
        let outcome = compile_quotient(
            &search_problem,
            &subgroup.restricted_action,
            stage_budget,
        )?;
        let CompileOutcomeV1::Authorized(certificate) = outcome else {
            return Err(QuotientErrorV1::TaskSemanticsMismatch);
        };
        environment_quotients.push(ContextualEnvironmentQuotientV1 {
            context_id: environment.environment_id.clone(),
            task_preserving_subgroup: subgroup,
            certificate,
        });
    }
    Ok(ContextualMultiEnvironmentCertificateV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        ambient_action_id: ambient_action.action_id.clone(),
        environment_quotients,
    })
}

/// Replay every environment-local subgroup and quotient. Context ordering and
/// exact context IDs are checked; using one context's certificate for another
/// environment is rejected rather than silently reinterpreted.
pub fn verify_contextual_multi_environment_quotient(
    problem: &MultiEnvironmentProblemV1,
    ambient_action: &FiniteActionV1,
    certificate: &ContextualMultiEnvironmentCertificateV1,
    budget: QualificationBudgetV1,
) -> Result<ContextualMultiEnvironmentVerificationReceiptV1, QuotientErrorV1> {
    if certificate.schema_version != 1
        || certificate.problem_id != problem.problem_id
        || certificate.ambient_action_id != ambient_action.action_id
        || certificate.environment_quotients.len() != problem.environments.len()
    {
        return Err(QuotientErrorV1::EnvironmentCertificateMismatch);
    }
    let expected = compile_contextual_multi_environment_quotient(
        problem,
        ambient_action,
        budget,
    )?;
    if *certificate != expected {
        return Err(QuotientErrorV1::EnvironmentCertificateMismatch);
    }
    let stage_budget = contextual_stage_budget(problem, budget)?;
    let mut environments = Vec::with_capacity(problem.environments.len());
    for (environment, quotient) in problem
        .environments
        .iter()
        .zip(certificate.environment_quotients.iter())
    {
        if quotient.context_id != environment.environment_id
            || quotient.task_preserving_subgroup.preserving_environment_ids
                != vec![environment.environment_id.clone()]
        {
            return Err(QuotientErrorV1::EnvironmentCertificateMismatch);
        }
        let search_problem = SearchProblemV1 {
            problem_id: format!("{}::{}", problem.problem_id, environment.environment_id),
            candidate_ids: problem.candidate_ids.clone(),
            semantics: environment.semantics.clone(),
        };
        let receipt = verify_certificate(
            &search_problem,
            &quotient.task_preserving_subgroup.restricted_action,
            &quotient.certificate,
            stage_budget,
        )?;
        environments.push(ContextualEnvironmentVerificationV1 {
            context_id: environment.environment_id.clone(),
            preserving_subgroup_order: quotient
                .task_preserving_subgroup
                .restricted_action
                .group
                .multiplication
                .len(),
            orbit_count: receipt.orbit_count,
        });
    }
    Ok(ContextualMultiEnvironmentVerificationReceiptV1 {
        problem_id: problem.problem_id.clone(),
        ambient_group_order: ambient_action.group.multiplication.len(),
        candidate_count: problem.candidate_ids.len(),
        environments,
    })
}

/// Independently reconstruct the common subgroup and every environment
/// quotient. Mutated environment semantics, subgroup witnesses, exclusions,
/// orbit certificates, or environment ordering are rejected.
pub fn verify_multi_environment_quotient(
    problem: &MultiEnvironmentProblemV1,
    ambient_action: &FiniteActionV1,
    certificate: &MultiEnvironmentQuotientCertificateV1,
    budget: QualificationBudgetV1,
) -> Result<MultiEnvironmentVerificationReceiptV1, QuotientErrorV1> {
    if certificate.schema_version != 1 || certificate.problem_id != problem.problem_id {
        return Err(QuotientErrorV1::EnvironmentCertificateMismatch);
    }
    let expected = compile_multi_environment_quotient(problem, ambient_action, budget)?;
    if *certificate != expected {
        return Err(QuotientErrorV1::EnvironmentCertificateMismatch);
    }

    let stage_budget = multi_environment_stage_budget(problem, budget)?;
    let mut per_environment_orbit_counts = Vec::with_capacity(problem.environments.len());
    for (environment, quotient) in problem
        .environments
        .iter()
        .zip(certificate.environment_quotients.iter())
    {
        let search_problem = SearchProblemV1 {
            problem_id: format!("{}::{}", problem.problem_id, environment.environment_id),
            candidate_ids: problem.candidate_ids.clone(),
            semantics: environment.semantics.clone(),
        };
        let receipt = verify_certificate(
            &search_problem,
            &certificate.common_subgroup.restricted_action,
            &quotient.certificate,
            stage_budget,
        )?;
        per_environment_orbit_counts.push((environment.environment_id.clone(), receipt.orbit_count));
    }

    Ok(MultiEnvironmentVerificationReceiptV1 {
        problem_id: problem.problem_id.clone(),
        environment_count: problem.environments.len(),
        ambient_group_order: ambient_action.group.multiplication.len(),
        common_subgroup_order: certificate.common_subgroup.restricted_action.group.multiplication.len(),
        candidate_count: problem.candidate_ids.len(),
        per_environment_orbit_counts,
    })
}


/// Optimization sense is explicit because the certified representative search
/// must be interpreted against the original finite objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveSenseV1 {
    Minimize,
    Maximize,
}

/// This authority is intentionally disjoint from exact quotient authority.
/// It permits only an additive-objective-gap claim; it is not exact equivalence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApproximatePruningAuthorityV1 {
    ExactFeasibilityWithBoundedAdditiveObjectiveDeviation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApproximateSearchContractV1 {
    /// Unit/scale identifier for objective_cost and the declared tolerance.
    pub objective_unit_id: String,
    pub objective_sense: ObjectiveSenseV1,
    /// User-declared maximum acceptable additive objective gap in objective units.
    pub maximum_additive_gap: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApproximateBlockedV1 {
    HardTaskSemanticMismatch {
        group_element: usize,
        source_index: usize,
        image_index: usize,
        source_candidate_id: String,
        image_candidate_id: String,
        differing_fields: Vec<TaskFieldV1>,
    },
    ObjectiveDeviationExceedsTolerance {
        /// Group element mapping source_candidate_id to image_candidate_id.
        group_element: usize,
        source_index: usize,
        image_index: usize,
        source_candidate_id: String,
        image_candidate_id: String,
        /// Exact max-minus-min objective range within this orbit.
        absolute_objective_deviation: u64,
        maximum_additive_gap: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApproximateCompileOutcomeV1 {
    Authorized(ApproximateQuotientCertificateV1),
    Blocked(ApproximateBlockedV1),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApproximateMemberWitnessV1 {
    pub candidate_index: usize,
    pub candidate_id: String,
    pub group_element: usize,
    pub inverse_group_element: usize,
    /// Original objective value, bound to the complete problem contract.
    pub objective_value: i64,
    /// Exact absolute difference from this orbit's canonical representative.
    pub objective_deviation_from_representative: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApproximateOrbitV1 {
    pub representative_index: usize,
    pub representative_id: String,
    pub members: Vec<ApproximateMemberWitnessV1>,
    pub stabilizer_group_elements: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApproximateCertificateEvidenceV1 {
    pub group_axiom_checks: usize,
    pub action_law_checks: usize,
    pub exact_non_objective_semantic_checks: usize,
    pub member_objective_checks: usize,
    pub pairwise_objective_range_checks: usize,
    pub orbit_member_witnesses: usize,
    pub complete_domain_cover: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApproximateQuotientCertificateV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub action_id: String,
    pub group_id: String,
    pub group_order: usize,
    pub candidate_count: usize,
    pub pruning_authority: ApproximatePruningAuthorityV1,
    pub objective_unit_id: String,
    pub objective_sense: ObjectiveSenseV1,
    pub requested_maximum_additive_gap: u64,
    /// Exact, versioned encodings of the complete original inputs.
    pub problem_contract_bytes: Vec<u8>,
    pub action_contract_bytes: Vec<u8>,
    /// Maximum objective range max(f)-min(f) over any one orbit.
    pub maximum_pairwise_objective_deviation: u64,
    /// Verified max |f(candidate) - f(canonical orbit representative)|.
    pub certified_additive_optimality_gap: u64,
    pub orbits: Vec<ApproximateOrbitV1>,
    pub evidence: ApproximateCertificateEvidenceV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApproximateQuotientErrorV1 {
    Base(QuotientErrorV1),
    InvalidObjectiveUnit,
    Blocked(ApproximateBlockedV1),
    CertificateMismatch,
    NoFeasibleRepresentative,
    SearchResultMismatch,
}

impl fmt::Display for ApproximateQuotientErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ApproximateQuotientErrorV1 {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApproximateVerificationReceiptV1 {
    pub problem_id: String,
    pub action_id: String,
    pub objective_unit_id: String,
    pub objective_sense: ObjectiveSenseV1,
    pub candidate_count: usize,
    pub orbit_count: usize,
    pub witnessed_member_count: usize,
    pub exact_non_objective_semantic_checks: usize,
    pub member_objective_checks: usize,
    pub maximum_pairwise_objective_deviation: u64,
    pub certified_additive_optimality_gap: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApproximateSearchResultV1 {
    pub problem_id: String,
    pub objective_unit_id: String,
    pub objective_sense: ObjectiveSenseV1,
    pub selected_candidate_index: usize,
    pub selected_candidate_id: String,
    pub selected_objective_value: i64,
    pub candidate_count: usize,
    pub representative_count: usize,
    pub requested_maximum_additive_gap: u64,
    pub certified_additive_optimality_gap: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApproximateSearchVerificationReceiptV1 {
    pub problem_id: String,
    pub selected_candidate_id: String,
    pub selected_objective_value: i64,
    pub full_candidate_count: usize,
    pub representative_count: usize,
    pub certified_additive_optimality_gap: u64,
}

fn append_contract_length(bytes: &mut Vec<u8>, length: usize) {
    bytes.extend_from_slice(&(length as u64).to_le_bytes());
}

fn append_contract_string(bytes: &mut Vec<u8>, value: &str) {
    append_contract_length(bytes, value.len());
    bytes.extend_from_slice(value.as_bytes());
}

/// Canonical encoding for exact replay identity. This is not a cryptographic
/// digest or signature; every original field is retained in versioned bytes.
fn canonical_problem_contract_bytes(problem: &SearchProblemV1) -> Vec<u8> {
    let mut bytes = b"luminous.symmetry.error_budgeted.problem.v1\0".to_vec();
    append_contract_string(&mut bytes, &problem.problem_id);
    append_contract_length(&mut bytes, problem.candidate_ids.len());
    for candidate_id in &problem.candidate_ids {
        append_contract_string(&mut bytes, candidate_id);
    }
    append_contract_length(&mut bytes, problem.semantics.len());
    for semantics in &problem.semantics {
        bytes.push(u8::from(semantics.in_search_domain));
        bytes.push(u8::from(semantics.admissible));
        bytes.push(u8::from(semantics.feasible));
        bytes.extend_from_slice(&semantics.objective_cost.to_le_bytes());
        append_contract_length(&mut bytes, semantics.constraint_signature.len());
        for value in &semantics.constraint_signature {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        append_contract_string(&mut bytes, &semantics.boundary_condition_id);
        append_contract_string(&mut bytes, &semantics.output_equivalence_id);
    }
    bytes
}

fn canonical_action_contract_bytes(action: &FiniteActionV1) -> Vec<u8> {
    let mut bytes = b"luminous.symmetry.error_budgeted.action.v1\\0".to_vec();
    append_contract_string(&mut bytes, &action.action_id);
    append_contract_string(&mut bytes, &action.group.group_id);
    append_contract_length(&mut bytes, action.group.identity);
    append_contract_length(&mut bytes, action.group.multiplication.len());
    for row in &action.group.multiplication {
        append_contract_length(&mut bytes, row.len());
        for product in row {
            append_contract_length(&mut bytes, *product);
        }
    }
    append_contract_length(&mut bytes, action.permutations.len());
    for permutation in &action.permutations {
        append_contract_length(&mut bytes, permutation.len());
        for image in permutation {
            append_contract_length(&mut bytes, *image);
        }
    }
    bytes
}

fn non_objective_task_differences(
    source: &CandidateSemanticsV1,
    image: &CandidateSemanticsV1,
) -> Vec<TaskFieldV1> {
    let mut differences = Vec::new();
    if source.in_search_domain != image.in_search_domain {
        differences.push(TaskFieldV1::InSearchDomain);
    }
    if source.admissible != image.admissible {
        differences.push(TaskFieldV1::Admissible);
    }
    if source.feasible != image.feasible {
        differences.push(TaskFieldV1::Feasible);
    }
    if source.constraint_signature != image.constraint_signature {
        differences.push(TaskFieldV1::ConstraintSignature);
    }
    if source.boundary_condition_id != image.boundary_condition_id {
        differences.push(TaskFieldV1::BoundaryConditionId);
    }
    if source.output_equivalence_id != image.output_equivalence_id {
        differences.push(TaskFieldV1::OutputEquivalenceId);
    }
    differences
}

/// Compile a finite quotient when every non-objective task semantic is exact
/// but the scalar objective may vary within each orbit by a declared additive
/// tolerance. This is NOT an exact quotient and emits a distinct authority.
///
/// The error bound is relative to the deterministic canonical representative
/// of each orbit. The best representative is therefore epsilon-optimal for
/// either minimization or maximization over the original finite feasible set.
pub fn compile_approximate_quotient(
    problem: &SearchProblemV1,
    action: &FiniteActionV1,
    contract: &ApproximateSearchContractV1,
    budget: QualificationBudgetV1,
) -> Result<ApproximateCompileOutcomeV1, ApproximateQuotientErrorV1> {
    if !nonempty(&contract.objective_unit_id) {
        return Err(ApproximateQuotientErrorV1::InvalidObjectiveUnit);
    }
    let group_order = action.group.multiplication.len();
    let candidate_count = problem.candidate_ids.len();
    preflight(group_order, candidate_count, budget)
        .map_err(ApproximateQuotientErrorV1::Base)?;
    validate_problem(problem, budget).map_err(ApproximateQuotientErrorV1::Base)?;
    let validated =
        validate_action(action, candidate_count, budget).map_err(ApproximateQuotientErrorV1::Base)?;

    let mut exact_semantic_checks = 0usize;
    for (element, permutation) in action.permutations.iter().enumerate() {
        for (source_index, image_index) in permutation.iter().copied().enumerate() {
            exact_semantic_checks = exact_semantic_checks
                .checked_add(1)
                .ok_or(ApproximateQuotientErrorV1::Base(
                    QuotientErrorV1::ArithmeticOverflow,
                ))?;
            let differing_fields = non_objective_task_differences(
                &problem.semantics[source_index],
                &problem.semantics[image_index],
            );
            if !differing_fields.is_empty() {
                return Ok(ApproximateCompileOutcomeV1::Blocked(
                    ApproximateBlockedV1::HardTaskSemanticMismatch {
                        group_element: element,
                        source_index,
                        image_index,
                        source_candidate_id: problem.candidate_ids[source_index].clone(),
                        image_candidate_id: problem.candidate_ids[image_index].clone(),
                        differing_fields,
                    },
                ));
            }
        }
    }

    let problem_contract_bytes = canonical_problem_contract_bytes(problem);
    let action_contract_bytes = canonical_action_contract_bytes(action);
    let exact_orbits = build_orbits(problem, action, &validated.group.inverses);
    let mut orbits = Vec::with_capacity(exact_orbits.len());
    let mut member_objective_checks = 0usize;
    let mut pairwise_objective_range_checks = 0usize;
    let mut certified_gap = 0_u64;
    let mut maximum_pairwise_objective_deviation = 0_u64;

    for orbit in exact_orbits {
        let representative_index = orbit.representative_index;
        let representative_objective = problem.semantics[representative_index].objective_cost;
        let mut members = Vec::with_capacity(orbit.members.len());
        let mut minimum_member: Option<(i64, MemberWitnessV1)> = None;
        let mut maximum_member: Option<(i64, MemberWitnessV1)> = None;

        for member in orbit.members {
            member_objective_checks = member_objective_checks
                .checked_add(1)
                .ok_or(ApproximateQuotientErrorV1::Base(
                    QuotientErrorV1::ArithmeticOverflow,
                ))?;
            let objective_value = problem.semantics[member.candidate_index].objective_cost;
            let objective_deviation = representative_objective.abs_diff(objective_value);
            certified_gap = certified_gap.max(objective_deviation);

            if minimum_member
                .as_ref()
                .is_none_or(|(value, witness)| {
                    objective_value < *value
                        || (objective_value == *value
                            && member.candidate_id < witness.candidate_id)
                })
            {
                minimum_member = Some((objective_value, member.clone()));
            }
            if maximum_member
                .as_ref()
                .is_none_or(|(value, witness)| {
                    objective_value > *value
                        || (objective_value == *value
                            && member.candidate_id < witness.candidate_id)
                })
            {
                maximum_member = Some((objective_value, member.clone()));
            }

            members.push(ApproximateMemberWitnessV1 {
                candidate_index: member.candidate_index,
                candidate_id: member.candidate_id,
                group_element: member.group_element,
                inverse_group_element: member.inverse_group_element,
                objective_value,
                objective_deviation_from_representative: objective_deviation,
            });
        }

        pairwise_objective_range_checks = pairwise_objective_range_checks
            .checked_add(1)
            .ok_or(ApproximateQuotientErrorV1::Base(
                QuotientErrorV1::ArithmeticOverflow,
            ))?;
        let (minimum_value, minimum_witness) =
            minimum_member.expect("finite quotient orbits are non-empty");
        let (maximum_value, maximum_witness) =
            maximum_member.expect("finite quotient orbits are non-empty");
        let pairwise_deviation = minimum_value.abs_diff(maximum_value);
        maximum_pairwise_objective_deviation =
            maximum_pairwise_objective_deviation.max(pairwise_deviation);

        if pairwise_deviation > contract.maximum_additive_gap {
            // If g_min maps the representative to the minimum member and
            // g_max maps it to the maximum member, then
            // g_max * inverse(g_min) maps minimum -> maximum.
            let inverse_minimum_element =
                validated.group.inverses[minimum_witness.group_element];
            let group_element = action.group.multiplication[maximum_witness.group_element]
                [inverse_minimum_element];
            if action.permutations[group_element][minimum_witness.candidate_index]
                != maximum_witness.candidate_index
            {
                return Err(ApproximateQuotientErrorV1::Base(
                    QuotientErrorV1::ActionLawViolation,
                ));
            }
            return Ok(ApproximateCompileOutcomeV1::Blocked(
                ApproximateBlockedV1::ObjectiveDeviationExceedsTolerance {
                    group_element,
                    source_index: minimum_witness.candidate_index,
                    image_index: maximum_witness.candidate_index,
                    source_candidate_id: minimum_witness.candidate_id,
                    image_candidate_id: maximum_witness.candidate_id,
                    absolute_objective_deviation: pairwise_deviation,
                    maximum_additive_gap: contract.maximum_additive_gap,
                },
            ));
        }

        orbits.push(ApproximateOrbitV1 {
            representative_index,
            representative_id: orbit.representative_id,
            members,
            stabilizer_group_elements: orbit.stabilizer_group_elements,
        });
    }

    Ok(ApproximateCompileOutcomeV1::Authorized(
        ApproximateQuotientCertificateV1 {
            schema_version: 1,
            problem_id: problem.problem_id.clone(),
            action_id: action.action_id.clone(),
            group_id: action.group.group_id.clone(),
            group_order,
            candidate_count,
            pruning_authority:
                ApproximatePruningAuthorityV1::ExactFeasibilityWithBoundedAdditiveObjectiveDeviation,
            objective_unit_id: contract.objective_unit_id.clone(),
            objective_sense: contract.objective_sense,
            requested_maximum_additive_gap: contract.maximum_additive_gap,
            problem_contract_bytes,
            action_contract_bytes,
            maximum_pairwise_objective_deviation,
            certified_additive_optimality_gap: certified_gap,
            orbits,
            evidence: ApproximateCertificateEvidenceV1 {
                group_axiom_checks: validated.group.checks,
                action_law_checks: validated.checks,
                exact_non_objective_semantic_checks: exact_semantic_checks,
                member_objective_checks,
                pairwise_objective_range_checks,
                orbit_member_witnesses: candidate_count,
                complete_domain_cover: true,
            },
        },
    ))
}

/// Independently recompute the bounded-error quotient. The caller must supply
/// the original complete problem/action and the exact unit, sense and tolerance.
pub fn verify_approximate_certificate(
    problem: &SearchProblemV1,
    action: &FiniteActionV1,
    contract: &ApproximateSearchContractV1,
    certificate: &ApproximateQuotientCertificateV1,
    budget: QualificationBudgetV1,
) -> Result<ApproximateVerificationReceiptV1, ApproximateQuotientErrorV1> {
    if certificate.problem_contract_bytes != canonical_problem_contract_bytes(problem)
        || certificate.action_contract_bytes != canonical_action_contract_bytes(action)
    {
        return Err(ApproximateQuotientErrorV1::CertificateMismatch);
    }
    let expected = match compile_approximate_quotient(problem, action, contract, budget)? {
        ApproximateCompileOutcomeV1::Authorized(expected) => expected,
        ApproximateCompileOutcomeV1::Blocked(violation) => {
            return Err(ApproximateQuotientErrorV1::Blocked(violation));
        }
    };
    if *certificate != expected {
        return Err(ApproximateQuotientErrorV1::CertificateMismatch);
    }
    Ok(ApproximateVerificationReceiptV1 {
        problem_id: problem.problem_id.clone(),
        action_id: action.action_id.clone(),
        objective_unit_id: contract.objective_unit_id.clone(),
        objective_sense: contract.objective_sense,
        candidate_count: certificate.candidate_count,
        orbit_count: certificate.orbits.len(),
        witnessed_member_count: certificate.evidence.orbit_member_witnesses,
        exact_non_objective_semantic_checks: certificate
            .evidence
            .exact_non_objective_semantic_checks,
        member_objective_checks: certificate.evidence.member_objective_checks,
        maximum_pairwise_objective_deviation: certificate.maximum_pairwise_objective_deviation,
        certified_additive_optimality_gap: certificate.certified_additive_optimality_gap,
    })
}

/// Search only the canonical representatives after replaying the approximate
/// certificate. Ineligible candidates are excluded exactly; feasibility is
/// never approximated. Ties are resolved by candidate ID.
pub fn optimize_approximate_representatives(
    problem: &SearchProblemV1,
    action: &FiniteActionV1,
    contract: &ApproximateSearchContractV1,
    certificate: &ApproximateQuotientCertificateV1,
    budget: QualificationBudgetV1,
) -> Result<ApproximateSearchResultV1, ApproximateQuotientErrorV1> {
    verify_approximate_certificate(problem, action, contract, certificate, budget)?;

    let mut best: Option<(usize, i64)> = None;
    for orbit in &certificate.orbits {
        let index = orbit.representative_index;
        let semantics = &problem.semantics[index];
        if !(semantics.in_search_domain && semantics.admissible && semantics.feasible) {
            continue;
        }
        let replace = match best {
            None => true,
            Some((current_index, current_value)) => {
                let better_value = match contract.objective_sense {
                    ObjectiveSenseV1::Minimize => semantics.objective_cost < current_value,
                    ObjectiveSenseV1::Maximize => semantics.objective_cost > current_value,
                };
                better_value
                    || (semantics.objective_cost == current_value
                        && problem.candidate_ids[index] < problem.candidate_ids[current_index])
            }
        };
        if replace {
            best = Some((index, semantics.objective_cost));
        }
    }
    let (selected_index, selected_objective) =
        best.ok_or(ApproximateQuotientErrorV1::NoFeasibleRepresentative)?;
    Ok(ApproximateSearchResultV1 {
        problem_id: problem.problem_id.clone(),
        objective_unit_id: contract.objective_unit_id.clone(),
        objective_sense: contract.objective_sense,
        selected_candidate_index: selected_index,
        selected_candidate_id: problem.candidate_ids[selected_index].clone(),
        selected_objective_value: selected_objective,
        candidate_count: problem.candidate_ids.len(),
        representative_count: certificate.orbits.len(),
        requested_maximum_additive_gap: contract.maximum_additive_gap,
        certified_additive_optimality_gap: certificate.certified_additive_optimality_gap,
    })
}

pub fn verify_approximate_search_result(
    problem: &SearchProblemV1,
    action: &FiniteActionV1,
    contract: &ApproximateSearchContractV1,
    certificate: &ApproximateQuotientCertificateV1,
    result: &ApproximateSearchResultV1,
    budget: QualificationBudgetV1,
) -> Result<ApproximateSearchVerificationReceiptV1, ApproximateQuotientErrorV1> {
    let expected = optimize_approximate_representatives(
        problem,
        action,
        contract,
        certificate,
        budget,
    )?;
    if *result != expected {
        return Err(ApproximateQuotientErrorV1::SearchResultMismatch);
    }
    Ok(ApproximateSearchVerificationReceiptV1 {
        problem_id: result.problem_id.clone(),
        selected_candidate_id: result.selected_candidate_id.clone(),
        selected_objective_value: result.selected_objective_value,
        full_candidate_count: result.candidate_count,
        representative_count: result.representative_count,
        certified_additive_optimality_gap: result.certified_additive_optimality_gap,
    })
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

    fn c3_action() -> FiniteActionV1 {
        FiniteActionV1 {
            action_id: "cyclic-three-objective-action-v1".to_owned(),
            group: FiniteGroupV1 {
                group_id: "C3-objective-v1".to_owned(),
                identity: 0,
                multiplication: vec![vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1]],
            },
            permutations: vec![vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1]],
        }
    }

    fn c3_pairwise_drift_problem() -> SearchProblemV1 {
        SearchProblemV1 {
            problem_id: "three-member-objective-range-v1".to_owned(),
            candidate_ids: vec!["C0".to_owned(), "C1".to_owned(), "C2".to_owned()],
            semantics: vec![
                semantics(10, 7, "same-boundary", "same-output"),
                semantics(12, 7, "same-boundary", "same-output"),
                semantics(8, 7, "same-boundary", "same-output"),
            ],
        }
    }

    fn c3_objective_action() -> FiniteActionV1 {
        FiniteActionV1 {
            action_id: "C3-objective-action-v1".to_owned(),
            group: FiniteGroupV1 {
                group_id: "C3-objective-group-v1".to_owned(),
                identity: 0,
                multiplication: vec![
                    vec![0, 1, 2],
                    vec![1, 2, 0],
                    vec![2, 0, 1],
                ],
            },
            permutations: vec![
                vec![0, 1, 2],
                vec![1, 2, 0],
                vec![2, 0, 1],
            ],
        }
    }

    fn c3_objective_problem() -> SearchProblemV1 {
        SearchProblemV1 {
            problem_id: "three-member-objective-range-v1".to_owned(),
            candidate_ids: vec!["C0".to_owned(), "C1".to_owned(), "C2".to_owned()],
            semantics: vec![
                semantics(10, 7, "same-boundary", "same-output"),
                semantics(12, 7, "same-boundary", "same-output"),
                semantics(8, 7, "same-boundary", "same-output"),
            ],
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


    fn d4_bounded_objective_problem() -> SearchProblemV1 {
        let mut candidate_ids = Vec::with_capacity(16);
        let mut semantics = Vec::with_capacity(16);
        for source in 0_usize..4 {
            for target in 0_usize..4 {
                let distance = source.abs_diff(target).min(4 - source.abs_diff(target));
                let base = match distance {
                    0 => 10_i64,
                    1 => 20_i64,
                    2 => 30_i64,
                    _ => unreachable!(),
                };
                let offset = match distance {
                    0 | 2 => (source % 3) as i64,
                    1 => ((source + target) % 3) as i64,
                    _ => unreachable!(),
                };
                candidate_ids.push(format!("p{source}{target}"));
                semantics.push(CandidateSemanticsV1 {
                    in_search_domain: true,
                    admissible: true,
                    feasible: true,
                    objective_cost: base + offset,
                    constraint_signature: vec![distance as i64],
                    boundary_condition_id: "fixed-d4-boundary".to_owned(),
                    output_equivalence_id: format!("distance-{distance}"),
                });
            }
        }
        SearchProblemV1 {
            problem_id: "D4-bounded-objective-drift-v1".to_owned(),
            candidate_ids,
            semantics,
        }
    }

    fn approximate_min_problem() -> SearchProblemV1 {
        let mut problem = c2_problem();
        problem.semantics[1].objective_cost = 8;
        problem.semantics[3].objective_cost = 22;
        problem
    }

    fn approximate_contract(
        sense: ObjectiveSenseV1,
        tolerance: u64,
    ) -> ApproximateSearchContractV1 {
        ApproximateSearchContractV1 {
            objective_unit_id: "microjoule-v1".to_owned(),
            objective_sense: sense,
            maximum_additive_gap: tolerance,
        }
    }

    #[test]
    fn bounded_objective_quotient_returns_a_verified_additive_regret_bound() {
        let problem = approximate_min_problem();
        let action = c2_action();
        let contract = approximate_contract(ObjectiveSenseV1::Minimize, 2);
        let outcome = compile_approximate_quotient(
            &problem,
            &action,
            &contract,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        let ApproximateCompileOutcomeV1::Authorized(certificate) = outcome else {
            panic!("expected an epsilon-authorized quotient");
        };
        assert_eq!(
            certificate.pruning_authority,
            ApproximatePruningAuthorityV1::ExactFeasibilityWithBoundedAdditiveObjectiveDeviation
        );
        assert_eq!(certificate.certified_additive_optimality_gap, 2);
        assert_eq!(certificate.requested_maximum_additive_gap, 2);
        assert_eq!(certificate.orbits.len(), 2);
        verify_approximate_certificate(
            &problem,
            &action,
            &contract,
            &certificate,
            QualificationBudgetV1::default(),
        )
        .unwrap();

        let result = optimize_approximate_representatives(
            &problem,
            &action,
            &contract,
            &certificate,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(result.selected_candidate_id, "A0");
        assert_eq!(result.selected_objective_value, 10);
        let exhaustive_optimum = problem
            .candidate_ids
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                let semantics = &problem.semantics[*index];
                semantics.in_search_domain && semantics.admissible && semantics.feasible
            })
            .map(|(index, _)| (index, problem.semantics[index].objective_cost))
            .min_by_key(|(_, value)| *value)
            .unwrap()
            .1;
        assert_eq!(result.selected_objective_value - exhaustive_optimum, 2);
        assert_eq!(
            verify_approximate_search_result(
                &problem,
                &action,
                &contract,
                &certificate,
                &result,
                QualificationBudgetV1::default(),
            )
            .unwrap()
            .certified_additive_optimality_gap,
            2
        );
    }

    #[test]
    fn approximate_quotient_has_the_same_additive_guarantee_for_maximization() {
        let mut problem = c2_problem();
        problem.semantics[1].objective_cost = 12;
        problem.semantics[3].objective_cost = 22;
        let action = c2_action();
        let contract = approximate_contract(ObjectiveSenseV1::Maximize, 2);
        let ApproximateCompileOutcomeV1::Authorized(certificate) =
            compile_approximate_quotient(
                &problem,
                &action,
                &contract,
                QualificationBudgetV1::default(),
            )
            .unwrap()
        else {
            panic!("expected bounded max quotient");
        };
        let result = optimize_approximate_representatives(
            &problem,
            &action,
            &contract,
            &certificate,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(result.selected_candidate_id, "B0");
        assert_eq!(result.selected_objective_value, 20);
        let exhaustive_max = problem
            .semantics
            .iter()
            .enumerate()
            .filter(|(_, semantics)| {
                semantics.in_search_domain && semantics.admissible && semantics.feasible
            })
            .map(|(_, semantics)| semantics.objective_cost)
            .max()
            .unwrap();
        assert_eq!(exhaustive_max - result.selected_objective_value, 2);
        assert_eq!(result.certified_additive_optimality_gap, 2);
    }

    #[test]
    fn objective_deviation_one_unit_over_tolerance_blocks_quotient() {
        let problem = approximate_min_problem();
        let action = c2_action();
        let contract = approximate_contract(ObjectiveSenseV1::Minimize, 1);
        let outcome = compile_approximate_quotient(
            &problem,
            &action,
            &contract,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert!(matches!(
            outcome,
            ApproximateCompileOutcomeV1::Blocked(
                ApproximateBlockedV1::ObjectiveDeviationExceedsTolerance {
                    absolute_objective_deviation: 2,
                    maximum_additive_gap: 1,
                    ..
                }
            )
        ));
    }

    #[test]
    fn any_feasibility_or_non_objective_semantic_change_blocks_even_with_large_tolerance() {
        let mut problem = approximate_min_problem();
        problem.semantics[1].feasible = false;
        let outcome = compile_approximate_quotient(
            &problem,
            &c2_action(),
            &approximate_contract(ObjectiveSenseV1::Minimize, u64::MAX),
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert!(matches!(
            outcome,
            ApproximateCompileOutcomeV1::Blocked(
                ApproximateBlockedV1::HardTaskSemanticMismatch {
                    differing_fields,
                    ..
                }
            ) if differing_fields == vec![TaskFieldV1::Feasible]
        ));

        let mut problem = approximate_min_problem();
        problem.semantics[1].boundary_condition_id = "different-boundary".to_owned();
        let outcome = compile_approximate_quotient(
            &problem,
            &c2_action(),
            &approximate_contract(ObjectiveSenseV1::Minimize, u64::MAX),
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert!(matches!(
            outcome,
            ApproximateCompileOutcomeV1::Blocked(
                ApproximateBlockedV1::HardTaskSemanticMismatch {
                    differing_fields,
                    ..
                }
            ) if differing_fields == vec![TaskFieldV1::BoundaryConditionId]
        ));
    }

    #[test]
    fn pairwise_objective_range_is_checked_not_only_representative_deviation() {
        let problem = c3_pairwise_drift_problem();
        let action = c3_action();
        let contract = approximate_contract(ObjectiveSenseV1::Minimize, 2);
        let outcome = compile_approximate_quotient(
            &problem,
            &action,
            &contract,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        let ApproximateCompileOutcomeV1::Blocked(
            ApproximateBlockedV1::ObjectiveDeviationExceedsTolerance {
                group_element,
                source_index,
                image_index,
                source_candidate_id,
                image_candidate_id,
                absolute_objective_deviation,
                maximum_additive_gap,
            },
        ) = outcome
        else {
            panic!("pairwise 12-to-8 drift must exceed the declared tolerance");
        };
        assert_eq!(source_candidate_id, "C2");
        assert_eq!(image_candidate_id, "C1");
        assert_eq!((source_index, image_index), (2, 1));
        assert_eq!(absolute_objective_deviation, 4);
        assert_eq!(maximum_additive_gap, 2);
        assert_eq!(action.permutations[group_element][source_index], image_index);
    }

    #[test]
    fn pairwise_range_and_representative_regret_are_reported_separately() {
        let problem = c3_pairwise_drift_problem();
        let action = c3_action();
        let contract = approximate_contract(ObjectiveSenseV1::Minimize, 4);
        let ApproximateCompileOutcomeV1::Authorized(certificate) = compile_approximate_quotient(
            &problem,
            &action,
            &contract,
            QualificationBudgetV1::default(),
        )
        .unwrap()
        else {
            panic!("four-unit pairwise objective span should be authorized");
        };
        assert_eq!(certificate.maximum_pairwise_objective_deviation, 4);
        assert_eq!(certificate.certified_additive_optimality_gap, 2);
        assert_eq!(certificate.evidence.pairwise_objective_range_checks, 1);

        let result = optimize_approximate_representatives(
            &problem,
            &action,
            &contract,
            &certificate,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(result.selected_candidate_id, "C0");
        assert_eq!(result.selected_objective_value, 10);
        assert_eq!(result.selected_objective_value - 8, 2);
    }

    #[test]
    fn approximate_certificate_binds_full_problem_and_action_contracts() {
        let problem = approximate_min_problem();
        let action = c2_action();
        let contract = approximate_contract(ObjectiveSenseV1::Minimize, 2);
        let ApproximateCompileOutcomeV1::Authorized(certificate) = compile_approximate_quotient(
            &problem,
            &action,
            &contract,
            QualificationBudgetV1::default(),
        )
        .unwrap()
        else {
            panic!("fixture must authorize");
        };

        let mut shifted_problem = problem.clone();
        for semantics in &mut shifted_problem.semantics {
            semantics.objective_cost += 100;
        }
        assert_eq!(
            verify_approximate_certificate(
                &shifted_problem,
                &action,
                &contract,
                &certificate,
                QualificationBudgetV1::default(),
            ),
            Err(ApproximateQuotientErrorV1::CertificateMismatch)
        );

        let mut changed_action = action.clone();
        changed_action.permutations[1].swap(0, 1);
        assert_eq!(
            verify_approximate_certificate(
                &problem,
                &changed_action,
                &contract,
                &certificate,
                QualificationBudgetV1::default(),
            ),
            Err(ApproximateQuotientErrorV1::CertificateMismatch)
        );
    }

    #[test]
    fn pairwise_objective_range_blocks_three_member_orbit_hidden_by_representative_check() {
        let problem = c3_objective_problem();
        let action = c3_objective_action();
        let contract = approximate_contract(ObjectiveSenseV1::Minimize, 2);
        let outcome = compile_approximate_quotient(
            &problem,
            &action,
            &contract,
            QualificationBudgetV1::default(),
        )
        .unwrap();

        let ApproximateCompileOutcomeV1::Blocked(
            ApproximateBlockedV1::ObjectiveDeviationExceedsTolerance {
                group_element,
                source_index,
                image_index,
                source_candidate_id,
                image_candidate_id,
                absolute_objective_deviation,
                maximum_additive_gap,
            },
        ) = outcome
        else {
            panic!("pairwise objective range 12-8 must exceed tolerance 2");
        };

        assert_eq!(source_index, 2);
        assert_eq!(image_index, 1);
        assert_eq!(source_candidate_id, "C2");
        assert_eq!(image_candidate_id, "C1");
        assert_eq!(absolute_objective_deviation, 4);
        assert_eq!(maximum_additive_gap, 2);
        assert_eq!(action.permutations[group_element][source_index], image_index);
    }

    #[test]
    fn pairwise_range_and_representative_regret_are_certified_separately() {
        let problem = c3_objective_problem();
        let action = c3_objective_action();
        let contract = approximate_contract(ObjectiveSenseV1::Minimize, 4);
        let ApproximateCompileOutcomeV1::Authorized(certificate) = compile_approximate_quotient(
            &problem,
            &action,
            &contract,
            QualificationBudgetV1::default(),
        )
        .unwrap()
        else {
            panic!("pairwise range exactly at tolerance must be authorized");
        };

        assert_eq!(certificate.maximum_pairwise_objective_deviation, 4);
        assert_eq!(certificate.certified_additive_optimality_gap, 2);
        assert_eq!(certificate.evidence.pairwise_objective_range_checks, 1);
        assert_eq!(certificate.evidence.member_objective_checks, 3);

        verify_approximate_certificate(
            &problem,
            &action,
            &contract,
            &certificate,
            QualificationBudgetV1::default(),
        )
        .unwrap();

        let result = optimize_approximate_representatives(
            &problem,
            &action,
            &contract,
            &certificate,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(result.selected_candidate_id, "C0");
        assert_eq!(result.selected_objective_value, 10);
        assert_eq!(result.selected_objective_value - 8, 2);
    }

    #[test]
    fn approximate_certificate_binds_full_objective_and_action_inputs() {
        let problem = approximate_min_problem();
        let action = c2_action();
        let contract = approximate_contract(ObjectiveSenseV1::Minimize, 2);
        let ApproximateCompileOutcomeV1::Authorized(certificate) = compile_approximate_quotient(
            &problem,
            &action,
            &contract,
            QualificationBudgetV1::default(),
        )
        .unwrap()
        else {
            panic!("fixture must authorize");
        };

        // A uniform objective offset preserves all deviations but changes the
        // original task; the serialized original problem must therefore bind.
        let mut shifted_problem = problem.clone();
        for semantics in &mut shifted_problem.semantics {
            semantics.objective_cost += 100;
        }
        assert_eq!(
            verify_approximate_certificate(
                &shifted_problem,
                &action,
                &contract,
                &certificate,
                QualificationBudgetV1::default(),
            ),
            Err(ApproximateQuotientErrorV1::CertificateMismatch)
        );

        let mut changed_action = action.clone();
        changed_action.action_id = "mutated-action-identity".to_owned();
        assert_eq!(
            verify_approximate_certificate(
                &problem,
                &changed_action,
                &contract,
                &certificate,
                QualificationBudgetV1::default(),
            ),
            Err(ApproximateQuotientErrorV1::CertificateMismatch)
        );
    }

    #[test]
    fn non_abelian_d4_pairwise_objective_range_is_qualified_with_three_orbits() {
        let problem = d4_bounded_objective_problem();
        let action = d4_action();
        // Prove this is not an abelian smoke test.
        assert_ne!(
            action.group.multiplication[1][4],
            action.group.multiplication[4][1]
        );
        let contract = approximate_contract(ObjectiveSenseV1::Minimize, 2);
        let ApproximateCompileOutcomeV1::Authorized(certificate) = compile_approximate_quotient(
            &problem,
            &action,
            &contract,
            QualificationBudgetV1::default(),
        )
        .unwrap()
        else {
            panic!("D4 orbit ranges exactly at tolerance must qualify");
        };

        assert_eq!(certificate.orbits.len(), 3);
        assert_eq!(
            certificate
                .orbits
                .iter()
                .map(|orbit| orbit.members.len())
                .collect::<Vec<_>>(),
            vec![4, 8, 4]
        );
        assert_eq!(certificate.maximum_pairwise_objective_deviation, 2);
        assert!(certificate.certified_additive_optimality_gap <= 2);
        verify_approximate_certificate(
            &problem,
            &action,
            &contract,
            &certificate,
            QualificationBudgetV1::default(),
        )
        .unwrap();
    }

    #[test]
    fn non_abelian_d4_over_tolerance_witness_maps_minimum_member_to_maximum() {
        let problem = d4_bounded_objective_problem();
        let action = d4_action();
        let contract = approximate_contract(ObjectiveSenseV1::Minimize, 1);
        let ApproximateCompileOutcomeV1::Blocked(
            ApproximateBlockedV1::ObjectiveDeviationExceedsTolerance {
                group_element,
                source_index,
                image_index,
                absolute_objective_deviation,
                maximum_additive_gap,
                ..
            },
        ) = compile_approximate_quotient(
            &problem,
            &action,
            &contract,
            QualificationBudgetV1::default(),
        )
        .unwrap()
        else {
            panic!("D4 objective range 2 must block at tolerance 1");
        };

        assert_eq!(absolute_objective_deviation, 2);
        assert_eq!(maximum_additive_gap, 1);
        assert_eq!(
            action.permutations[group_element][source_index],
            image_index
        );
        assert!(
            problem.semantics[source_index].objective_cost
                < problem.semantics[image_index].objective_cost
        );
    }

    #[test]
    fn epsilon_zero_matches_the_exact_quotient_and_is_not_promoted() {
        let problem = c2_problem();
        let action = c2_action();
        let exact_outcome =
            compile_quotient(&problem, &action, QualificationBudgetV1::default()).unwrap();
        let CompileOutcomeV1::Authorized(exact) = exact_outcome else {
            panic!("exact fixture should authorize the exact quotient");
        };
        let approximate_outcome = compile_approximate_quotient(
            &problem,
            &action,
            &approximate_contract(ObjectiveSenseV1::Minimize, 0),
            QualificationBudgetV1::default(),
        )
        .unwrap();
        let ApproximateCompileOutcomeV1::Authorized(approximate) = approximate_outcome else {
            panic!("zero-error quotient should be authorized");
        };
        assert_eq!(approximate.certified_additive_optimality_gap, 0);
        assert_eq!(
            exact
                .orbits
                .iter()
                .map(|orbit| orbit.representative_id.clone())
                .collect::<Vec<_>>(),
            approximate
                .orbits
                .iter()
                .map(|orbit| orbit.representative_id.clone())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            approximate.pruning_authority,
            ApproximatePruningAuthorityV1::ExactFeasibilityWithBoundedAdditiveObjectiveDeviation
        );
        assert_eq!(
            exact.pruning_authority,
            PruningAuthorityV1::ExactFiniteGroupAndExactTaskSemanticPreservation
        );
    }

    #[test]
    fn approximate_certificate_binds_unit_sense_tolerance_and_gap() {
        let problem = approximate_min_problem();
        let action = c2_action();
        let contract = approximate_contract(ObjectiveSenseV1::Minimize, 2);
        let ApproximateCompileOutcomeV1::Authorized(mut certificate) =
            compile_approximate_quotient(
                &problem,
                &action,
                &contract,
                QualificationBudgetV1::default(),
            )
            .unwrap()
        else {
            panic!("expected bounded quotient");
        };

        let mut forged = certificate.clone();
        forged.certified_additive_optimality_gap = 0;
        assert_eq!(
            verify_approximate_certificate(
                &problem,
                &action,
                &contract,
                &forged,
                QualificationBudgetV1::default(),
            ),
            Err(ApproximateQuotientErrorV1::CertificateMismatch)
        );

        assert!(matches!(
            verify_approximate_certificate(
                &problem,
                &action,
                &approximate_contract(ObjectiveSenseV1::Minimize, 3),
                &certificate,
                QualificationBudgetV1::default(),
            ),
            Err(ApproximateQuotientErrorV1::CertificateMismatch)
        ));
        assert!(matches!(
            verify_approximate_certificate(
                &problem,
                &action,
                &approximate_contract(ObjectiveSenseV1::Maximize, 2),
                &certificate,
                QualificationBudgetV1::default(),
            ),
            Err(ApproximateQuotientErrorV1::CertificateMismatch)
        ));
        let mut changed_unit = approximate_contract(ObjectiveSenseV1::Minimize, 2);
        changed_unit.objective_unit_id = "millijoule-v1".to_owned();
        assert!(matches!(
            verify_approximate_certificate(
                &problem,
                &action,
                &changed_unit,
                &certificate,
                QualificationBudgetV1::default(),
            ),
            Err(ApproximateQuotientErrorV1::CertificateMismatch)
        ));

        certificate.orbits[0].members[0].objective_deviation_from_representative = 99;
        assert_eq!(
            verify_approximate_certificate(
                &problem,
                &action,
                &contract,
                &certificate,
                QualificationBudgetV1::default(),
            ),
            Err(ApproximateQuotientErrorV1::CertificateMismatch)
        );
    }

    #[test]
    fn approximate_search_excludes_ineligible_candidates_exactly() {
        let mut problem = approximate_min_problem();
        problem.semantics[0].feasible = false;
        problem.semantics[1].feasible = false;
        problem.semantics[2].objective_cost = 30;
        problem.semantics[3].objective_cost = 30;
        let action = c2_action();
        let contract = approximate_contract(ObjectiveSenseV1::Minimize, 2);
        let ApproximateCompileOutcomeV1::Authorized(certificate) =
            compile_approximate_quotient(
                &problem,
                &action,
                &contract,
                QualificationBudgetV1::default(),
            )
            .unwrap()
        else {
            panic!("non-objective semantics remain constant within each orbit");
        };
        let result = optimize_approximate_representatives(
            &problem,
            &action,
            &contract,
            &certificate,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(result.selected_candidate_id, "B0");
    }

    #[test]
    fn approximate_work_budget_and_objective_unit_fail_closed() {
        let problem = approximate_min_problem();
        let action = c2_action();
        let mut invalid_unit = approximate_contract(ObjectiveSenseV1::Minimize, 2);
        invalid_unit.objective_unit_id.clear();
        assert_eq!(
            compile_approximate_quotient(
                &problem,
                &action,
                &invalid_unit,
                QualificationBudgetV1::default(),
            ),
            Err(ApproximateQuotientErrorV1::InvalidObjectiveUnit)
        );
        let tiny = QualificationBudgetV1 {
            maximum_work_units: 1,
            ..QualificationBudgetV1::default()
        };
        assert_eq!(
            compile_approximate_quotient(
                &problem,
                &action,
                &approximate_contract(ObjectiveSenseV1::Minimize, 2),
                tiny,
            ),
            Err(ApproximateQuotientErrorV1::Base(
                QuotientErrorV1::WorkBudgetExceeded
            ))
        );
    }

    #[test]
    fn intersection_of_environment_stabilizers_is_exact_common_subgroup() {
        let action = d4_action();
        let mut candidate_ids = Vec::with_capacity(16);
        let mut rotational_semantics = Vec::with_capacity(16);
        let mut operational_semantics = Vec::with_capacity(16);
        for source in 0usize..4 {
            for target in 0usize..4 {
                let delta = (target + 4 - source) % 4;
                let unoriented_distance = delta.min(4 - delta) as i64;
                let id = format!("p{source}{target}");
                candidate_ids.push(id);
                rotational_semantics.push(semantics(
                    unoriented_distance,
                    unoriented_distance,
                    "nominal-boundary",
                    "distance-output",
                ));
                operational_semantics.push(semantics(
                    delta as i64,
                    delta as i64,
                    "rotating-boundary",
                    "oriented-output",
                ));
            }
        }
        let problem = MultiEnvironmentProblemV1 {
            problem_id: "square-search-across-two-environments-v1".to_owned(),
            candidate_ids,
            environments: vec![
                TaskEnvironmentV1 {
                    environment_id: "nominal-isotropic".to_owned(),
                    semantics: rotational_semantics,
                },
                TaskEnvironmentV1 {
                    environment_id: "rotating-operating-regime".to_owned(),
                    semantics: operational_semantics,
                },
            ],
        };

        let subgroup =
            qualify_common_task_preserving_subgroup(
                &problem,
                &action,
                QualificationBudgetV1::default(),
            )
            .unwrap();
        assert_eq!(subgroup.retained_ambient_elements, vec![0, 1, 2, 3]);
        assert_eq!(subgroup.exclusion_witnesses.len(), 4);
        assert_eq!(
            subgroup
                .exclusion_witnesses
                .iter()
                .map(|witness| witness.ambient_group_element)
                .collect::<Vec<_>>(),
            vec![4, 5, 6, 7]
        );
        assert!(subgroup.exclusion_witnesses.iter().all(|witness| {
            witness.environment_id == "rotating-operating-regime"
                && witness.differing_fields.contains(&TaskFieldV1::ObjectiveCost)
        }));

        let certificate = compile_multi_environment_quotient(
            &problem,
            &action,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        let receipt = verify_multi_environment_quotient(
            &problem,
            &action,
            &certificate,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(receipt.environment_count, 2);
        assert_eq!(receipt.ambient_group_order, 8);
        assert_eq!(receipt.common_subgroup_order, 4);
        assert_eq!(receipt.candidate_count, 16);
        assert_eq!(
            receipt.per_environment_orbit_counts,
            vec![
                ("nominal-isotropic".to_owned(), 4),
                ("rotating-operating-regime".to_owned(), 4),
            ]
        );
    }

    #[test]
    fn third_environment_can_shrink_common_symmetry_to_identity() {
        let action = d4_action();
        let candidate_ids: Vec<String> = (0..16).map(|index| format!("p{index:02}")).collect();
        let d4_semantics = vec![semantics(1, 1, "base", "same"); 16];
        let c4_semantics = (0..16)
            .map(|index| {
                let source = index / 4;
                let target = index % 4;
                let delta = (target + 4 - source) % 4;
                semantics(delta as i64, 0, "base", "same")
            })
            .collect();
        let asymmetric_semantics = (0..16)
            .map(|index| semantics(index as i64, 0, "base", "same"))
            .collect();
        let problem = MultiEnvironmentProblemV1 {
            problem_id: "three-environment-symmetry-intersection-v1".to_owned(),
            candidate_ids,
            environments: vec![
                TaskEnvironmentV1 {
                    environment_id: "fully-symmetric".to_owned(),
                    semantics: d4_semantics,
                },
                TaskEnvironmentV1 {
                    environment_id: "rotation-only".to_owned(),
                    semantics: c4_semantics,
                },
                TaskEnvironmentV1 {
                    environment_id: "asymmetric-control".to_owned(),
                    semantics: asymmetric_semantics,
                },
            ],
        };
        let subgroup = qualify_common_task_preserving_subgroup(
            &problem,
            &action,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(subgroup.retained_ambient_elements, vec![0]);
        assert_eq!(subgroup.restricted_action.group.multiplication, vec![vec![0]]);
        let certificate = compile_multi_environment_quotient(
            &problem,
            &action,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        let receipt = verify_multi_environment_quotient(
            &problem,
            &action,
            &certificate,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(receipt.common_subgroup_order, 1);
        assert_eq!(receipt.per_environment_orbit_counts.len(), 3);
        assert!(receipt.per_environment_orbit_counts.iter().all(|entry| entry.1 == 16));
    }

    #[test]
    fn contextual_symmetry_atlas_preserves_each_environment_local_stabilizer() {
        let action = d4_action();
        let mut candidate_ids = Vec::with_capacity(16);
        let mut d4_semantics = Vec::with_capacity(16);
        let mut c4_semantics = Vec::with_capacity(16);
        let mut c1_semantics = Vec::with_capacity(16);
        for source in 0usize..4 {
            for target in 0usize..4 {
                let delta = (target + 4 - source) % 4;
                candidate_ids.push(format!("p{source}{target}"));
                d4_semantics.push(semantics(0, 0, "same-boundary", "same-output"));
                c4_semantics.push(semantics(
                    delta as i64,
                    delta as i64,
                    "rotating-boundary",
                    "oriented-output",
                ));
                c1_semantics.push(semantics(
                    (source * 4 + target) as i64,
                    0,
                    "asymmetric-boundary",
                    "unique-output",
                ));
            }
        }
        let problem = MultiEnvironmentProblemV1 {
            problem_id: "square-search-symmetry-atlas-v1".to_owned(),
            candidate_ids,
            environments: vec![
                TaskEnvironmentV1 {
                    environment_id: "isotropic".to_owned(),
                    semantics: d4_semantics,
                },
                TaskEnvironmentV1 {
                    environment_id: "rotating-regime".to_owned(),
                    semantics: c4_semantics,
                },
                TaskEnvironmentV1 {
                    environment_id: "asymmetric-regime".to_owned(),
                    semantics: c1_semantics,
                },
            ],
        };

        let atlas = compile_contextual_multi_environment_quotient(
            &problem,
            &action,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(
            atlas
                .environment_quotients
                .iter()
                .map(|entry| entry.task_preserving_subgroup.retained_ambient_elements.clone())
                .collect::<Vec<_>>(),
            vec![vec![0, 1, 2, 3, 4, 5, 6, 7], vec![0, 1, 2, 3], vec![0]]
        );
        let receipt = verify_contextual_multi_environment_quotient(
            &problem,
            &action,
            &atlas,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(
            receipt
                .environments
                .iter()
                .map(|entry| (entry.context_id.as_str(), entry.preserving_subgroup_order, entry.orbit_count))
                .collect::<Vec<_>>(),
            vec![
                ("isotropic", 8, 3),
                ("rotating-regime", 4, 4),
                ("asymmetric-regime", 1, 16),
            ]
        );
    }

    #[test]
    fn contextual_atlas_rejects_a_quotient_relabelled_for_another_environment() {
        let action = d4_action();
        let problem = MultiEnvironmentProblemV1 {
            problem_id: "context-bound-certificate-v1".to_owned(),
            candidate_ids: (0..16).map(|index| format!("p{index:02}")).collect(),
            environments: vec![
                TaskEnvironmentV1 {
                    environment_id: "context-a".to_owned(),
                    semantics: vec![semantics(1, 1, "boundary", "output"); 16],
                },
                TaskEnvironmentV1 {
                    environment_id: "context-b".to_owned(),
                    semantics: vec![semantics(2, 2, "boundary", "output"); 16],
                },
            ],
        };
        let mut atlas = compile_contextual_multi_environment_quotient(
            &problem,
            &action,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        atlas.environment_quotients[0].context_id = "context-b".to_owned();
        assert_eq!(
            verify_contextual_multi_environment_quotient(
                &problem,
                &action,
                &atlas,
                QualificationBudgetV1::default(),
            ),
            Err(QuotientErrorV1::EnvironmentCertificateMismatch)
        );
    }

    #[test]
    fn multi_environment_total_work_budget_is_shared_across_compile_and_replay() {
        let action = d4_action();
        let semantics_a = vec![semantics(1, 1, "boundary", "output"); 16];
        let semantics_b = vec![semantics(1, 1, "boundary", "output"); 16];
        let problem = MultiEnvironmentProblemV1 {
            problem_id: "multi-environment-budget-v1".to_owned(),
            candidate_ids: (0..16).map(|index| format!("p{index:02}")).collect(),
            environments: vec![
                TaskEnvironmentV1 {
                    environment_id: "env-a".to_owned(),
                    semantics: semantics_a,
                },
                TaskEnvironmentV1 {
                    environment_id: "env-b".to_owned(),
                    semantics: semantics_b,
                },
            ],
        };
        let budget = QualificationBudgetV1 {
            maximum_work_units: 10_000,
            ..QualificationBudgetV1::default()
        };
        assert_eq!(
            compile_multi_environment_quotient(&problem, &action, budget),
            Err(QuotientErrorV1::WorkBudgetExceeded)
        );
    }

    #[test]
    fn multi_environment_certificate_rejects_mutated_environment_and_witness() {
        let action = d4_action();
        let problem = MultiEnvironmentProblemV1 {
            problem_id: "mutated-environment-certificate-v1".to_owned(),
            candidate_ids: (0..16).map(|index| format!("p{index:02}")).collect(),
            environments: vec![TaskEnvironmentV1 {
                environment_id: "single-environment".to_owned(),
                semantics: vec![semantics(1, 1, "boundary", "output"); 16],
            }],
        };
        let mut certificate = compile_multi_environment_quotient(
            &problem,
            &action,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        verify_multi_environment_quotient(
            &problem,
            &action,
            &certificate,
            QualificationBudgetV1::default(),
        )
        .unwrap();

        certificate.common_subgroup.exclusion_witnesses.push(
            EnvironmentExclusionWitnessV1 {
                ambient_group_element: 7,
                environment_id: "forged".to_owned(),
                source_index: 0,
                image_index: 1,
                source_candidate_id: "p00".to_owned(),
                image_candidate_id: "p01".to_owned(),
                differing_fields: vec![TaskFieldV1::ObjectiveCost],
            },
        );
        assert_eq!(
            verify_multi_environment_quotient(
                &problem,
                &action,
                &certificate,
                QualificationBudgetV1::default(),
            ),
            Err(QuotientErrorV1::EnvironmentCertificateMismatch)
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
