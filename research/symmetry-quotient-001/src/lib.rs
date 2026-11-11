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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CertificateEvidenceV1 {
    pub group_axiom_checks: usize,
    pub action_law_checks: usize,
    pub task_semantic_checks: usize,
    pub orbit_member_witnesses: usize,
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
    let g2 = group_order.checked_mul(group_order).ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    let g3 = g2.checked_mul(group_order).ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    let gn = group_order.checked_mul(candidate_count).ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    let g2n = g2.checked_mul(candidate_count).ok_or(QuotientErrorV1::ArithmeticOverflow)?;
    let required = g3.checked_add(g2n)
        .and_then(|v| v.checked_add(gn))
        .and_then(|v| v.checked_add(candidate_count))
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
            let element = action.permutations.iter()
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
        orbits.push(OrbitV1 {
            representative_index: representative,
            representative_id: problem.candidate_ids[representative].clone(),
            members,
        });
    }
    orbits.sort_by(|a, b| a.representative_id.cmp(&b.representative_id));
    orbits
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
            let first_witness = action.permutations.iter()
                .position(|permutation| permutation[rep] == index)
                .ok_or(QuotientErrorV1::BadWitness)?;
            if member.group_element != first_witness {
                return Err(QuotientErrorV1::BadWitness);
            }
            if problem.semantics[rep] != problem.semantics[index] {
                return Err(QuotientErrorV1::TaskSemanticsMismatch);
            }
        }

        let minimum_id = actual_members.iter()
            .map(|index| problem.candidate_ids[*index].as_str())
            .min()
            .ok_or(QuotientErrorV1::EmptyOrbit)?;
        if minimum_id != orbit.representative_id {
            return Err(QuotientErrorV1::BadRepresentative);
        }
        let expected_members: BTreeSet<usize> = action.permutations.iter()
            .map(|permutation| permutation[rep])
            .collect();
        if expected_members != actual_members {
            return Err(QuotientErrorV1::IncompleteOrbit);
        }
    }
    if covered.iter().any(|value| !value) || witnessed != candidate_count {
        return Err(QuotientErrorV1::IncompleteDomainCover);
    }

    let expected_evidence = CertificateEvidenceV1 {
        group_axiom_checks: validated.group.checks,
        action_law_checks: validated.checks,
        task_semantic_checks: task_checks,
        orbit_member_witnesses: candidate_count,
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
    let orbit = certificate.orbits.iter()
        .find(|orbit| orbit.representative_index == representative_index)
        .ok_or(QuotientErrorV1::RequestedCandidateIsNotRepresentative)?;
    Ok(orbit.members.iter().map(|member| LiftedCandidateV1 {
        candidate_index: member.candidate_index,
        candidate_id: member.candidate_id.clone(),
        group_element_from_representative: member.group_element,
        inverse_group_element: member.inverse_group_element,
    }).collect())
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
            candidate_ids: vec!["A0".to_owned(), "A1".to_owned(), "B0".to_owned(), "B1".to_owned()],
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
            let orbit: BTreeSet<usize> = action.permutations.iter()
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
        let outcome = compile_quotient(&problem, &action, QualificationBudgetV1::default()).unwrap();
        let CompileOutcomeV1::Authorized(certificate) = outcome else {
            panic!("expected an exact task-preserving quotient");
        };
        (problem, action, certificate)
    }

    #[test]
    fn c2_quotient_matches_exhaustive_oracle_and_lifts_all_members() {
        let (problem, action, certificate) = authorized_fixture();
        let receipt = verify_certificate(&problem, &action, &certificate, QualificationBudgetV1::default()).unwrap();
        let compiled: Vec<BTreeSet<usize>> = certificate.orbits.iter()
            .map(|orbit| orbit.members.iter().map(|member| member.candidate_index).collect())
            .collect();
        assert_eq!(compiled, exhaustive_oracle(&action, problem.candidate_ids.len()));
        assert_eq!(receipt.orbit_count, 2);
        assert_eq!(receipt.witnessed_member_count, 4);
        let lifted = lift_representative_solution(&problem, &action, &certificate, 0, QualificationBudgetV1::default()).unwrap();
        assert_eq!(lifted.iter().map(|member| member.candidate_id.as_str()).collect::<Vec<_>>(), vec!["A0", "A1"]);
        assert_eq!(
            lift_representative_solution(&problem, &action, &certificate, 1, QualificationBudgetV1::default()),
            Err(QuotientErrorV1::RequestedCandidateIsNotRepresentative)
        );
    }

    #[test]
    fn boundary_condition_change_blocks_pruning() {
        let action = c2_action();
        let mut problem = c2_problem();
        problem.semantics[1].boundary_condition_id = "different-boundary".to_owned();
        let outcome = compile_quotient(&problem, &action, QualificationBudgetV1::default()).unwrap();
        assert_eq!(outcome, CompileOutcomeV1::Blocked(TaskInvarianceViolationV1 {
            group_element: 1,
            source_index: 0,
            image_index: 1,
            source_candidate_id: "A0".to_owned(),
            image_candidate_id: "A1".to_owned(),
        }));
    }

    #[test]
    fn non_latin_multiplication_table_is_rejected() {
        let mut action = c2_action();
        action.group.multiplication = vec![vec![0, 1], vec![1, 1]];
        assert_eq!(validate_action(&action, 4, QualificationBudgetV1::default()), Err(QuotientErrorV1::MultiplicationTableNotLatin));
    }

    #[test]
    fn non_bijective_action_is_rejected() {
        let mut action = c2_action();
        action.permutations[1] = vec![0, 0, 2, 3];
        assert_eq!(validate_action(&action, 4, QualificationBudgetV1::default()), Err(QuotientErrorV1::NotPermutation));
    }

    #[test]
    fn omitted_orbit_member_is_rejected() {
        let (problem, action, mut certificate) = authorized_fixture();
        certificate.orbits[0].members.pop();
        assert_eq!(verify_certificate(&problem, &action, &certificate, QualificationBudgetV1::default()), Err(QuotientErrorV1::IncompleteOrbit));
    }

    #[test]
    fn incorrect_inverse_witness_is_rejected() {
        let (problem, action, mut certificate) = authorized_fixture();
        certificate.orbits[0].members[1].inverse_group_element = 0;
        assert_eq!(verify_certificate(&problem, &action, &certificate, QualificationBudgetV1::default()), Err(QuotientErrorV1::BadInverseWitness));
    }

    #[test]
    fn trivial_group_produces_singleton_orbits() {
        let problem = SearchProblemV1 {
            problem_id: "trivial-action-search".to_owned(),
            candidate_ids: vec!["c".to_owned(), "a".to_owned(), "b".to_owned()],
            semantics: vec![semantics(3, 0, "bc", "oc"), semantics(1, 0, "ba", "oa"), semantics(2, 0, "bb", "ob")],
        };
        let action = FiniteActionV1 {
            action_id: "trivial-action".to_owned(),
            group: FiniteGroupV1 { group_id: "C1".to_owned(), identity: 0, multiplication: vec![vec![0]] },
            permutations: vec![vec![0, 1, 2]],
        };
        let outcome = compile_quotient(&problem, &action, QualificationBudgetV1::default()).unwrap();
        let CompileOutcomeV1::Authorized(certificate) = outcome else { panic!("expected trivial quotient"); };
        assert_eq!(certificate.orbits.iter().map(|orbit| orbit.representative_id.as_str()).collect::<Vec<_>>(), vec!["a", "b", "c"]);
        verify_certificate(&problem, &action, &certificate, QualificationBudgetV1::default()).unwrap();
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
        };
        let outcome = compile_quotient(&problem, &action, QualificationBudgetV1::default()).unwrap();
        let CompileOutcomeV1::Authorized(certificate) = outcome else {
            panic!("expected an exact C3 quotient");
        };
        assert_eq!(certificate.orbits.len(), 1);
        assert_eq!(certificate.orbits[0].members.len(), 3);
        verify_certificate(&problem, &action, &certificate, QualificationBudgetV1::default()).unwrap();
    }

    #[test]
    fn work_budget_fails_before_qualification() {
        let budget = QualificationBudgetV1 { maximum_group_order: 4, maximum_candidates: 4, maximum_work_units: 1 };
        assert_eq!(compile_quotient(&c2_problem(), &c2_action(), budget), Err(QuotientErrorV1::WorkBudgetExceeded));
    }

    #[test]
    fn action_law_violation_is_rejected() {
        let mut action = c2_action();
        action.permutations[1] = vec![1, 0, 2, 3];
        assert_eq!(validate_action(&action, 4, QualificationBudgetV1::default()), Err(QuotientErrorV1::ActionLawViolation));
    }

    #[test]
    fn certificate_cannot_be_replayed_against_changed_task_semantics() {
        let (_problem, action, certificate) = authorized_fixture();
        let mut changed_problem = c2_problem();
        changed_problem.semantics[2].objective_cost += 1;
        assert_eq!(verify_certificate(&changed_problem, &action, &certificate, QualificationBudgetV1::default()), Err(QuotientErrorV1::TaskSemanticsMismatch));
    }
}
