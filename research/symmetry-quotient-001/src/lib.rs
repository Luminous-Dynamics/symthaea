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
    validate_multi_environment_problem(problem, budget)?;

    let environment_checks = problem
        .environments
        .len()
        .checked_mul(group_order)
        .and_then(|value| value.checked_mul(candidate_count))
        .ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    let remaining_work = budget
        .maximum_work_units
        .checked_sub(environment_checks)
        .ok_or(QuotientErrorV1::WorkBudgetExceeded)?;
    let adjusted_budget = QualificationBudgetV1 {
        maximum_work_units: remaining_work,
        ..budget
    };
    preflight(group_order, candidate_count, adjusted_budget)?;
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
    validate_multi_environment_problem(problem, budget)?;
    let stage_budget = contextual_stage_budget(problem, budget)?;
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
