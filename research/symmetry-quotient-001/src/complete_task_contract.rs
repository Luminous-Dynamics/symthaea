// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! V2 wrapper for exact finite quotients with an explicit complete task contract.
//!
//! The V1 candidate semantics remain useful for bounded discrete examples.
//! This wrapper additionally binds the declared universe, objective,
//! constraints, units, coordinate chart, initial/boundary conditions,
//! observables, data/evaluator/source revisions and numerical execution policy.
//! It does not infer missing metadata and does not create physical authority.

use super::{
    CompileOutcomeV1, FiniteActionV1, LiftedCandidateV1, QualificationBudgetV1,
    QuotientCertificateV1, QuotientErrorV1, SearchProblemV1, TaskInvarianceViolationV1,
    VerificationReceiptV1, canonical_search_problem_bytes, compile_quotient,
    lift_representative_solution, verify_certificate,
};

fn append_bytes(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(&(value.len() as u64).to_le_bytes());
    target.extend_from_slice(value);
}

fn append_string(target: &mut Vec<u8>, value: &str) {
    append_bytes(target, value.as_bytes());
}

/// Metadata that completes the V1 per-candidate task semantics for an exact
/// finite benchmark or bounded scientific search.
///
/// Every field is mandatory. Use an explicit value such as "not-applicable"
/// when a policy does not apply; empty, "unknown", and "unspecified" values
/// fail closed. Revision strings are identities supplied by the caller; the
/// wrapper does not claim that they are content hashes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchProblemMetadataV1 {
    pub candidate_universe_revision: String,
    pub action_domain_id: String,
    pub objective_definition_id: String,
    pub constraint_schema_id: String,
    pub unit_system_id: String,
    pub coordinate_chart_id: String,
    pub initial_condition_id: String,
    pub boundary_condition_family_id: String,
    pub observable_map_id: String,
    pub dataset_revision: String,
    pub evaluator_revision: String,
    pub source_revision: String,
    pub solver_profile_id: String,
    pub precision_policy_id: String,
    pub resource_budget_policy_id: String,
    pub termination_policy_id: String,
    pub randomness_policy_id: String,
    pub intervention_family_id: String,
    pub measurement_model_id: String,
    pub noise_model_id: String,
    pub context_selection_policy_id: String,
}

impl SearchProblemMetadataV1 {
    pub const FIELD_COUNT: usize = 21;

    fn fields(&self) -> [&str; Self::FIELD_COUNT] {
        [
            &self.candidate_universe_revision,
            &self.action_domain_id,
            &self.objective_definition_id,
            &self.constraint_schema_id,
            &self.unit_system_id,
            &self.coordinate_chart_id,
            &self.initial_condition_id,
            &self.boundary_condition_family_id,
            &self.observable_map_id,
            &self.dataset_revision,
            &self.evaluator_revision,
            &self.source_revision,
            &self.solver_profile_id,
            &self.precision_policy_id,
            &self.resource_budget_policy_id,
            &self.termination_policy_id,
            &self.randomness_policy_id,
            &self.intervention_family_id,
            &self.measurement_model_id,
            &self.noise_model_id,
            &self.context_selection_policy_id,
        ]
    }

    pub fn validate(&self) -> Result<(), QuotientErrorV1> {
        for value in self.fields() {
            let normalized = value.trim().to_ascii_lowercase();
            if normalized.is_empty()
                || matches!(normalized.as_str(), "unknown" | "unspecified" | "tbd" | "todo")
            {
                return Err(QuotientErrorV1::InvalidIdentifier);
            }
        }
        Ok(())
    }

    /// Canonical versioned metadata bytes. This is a deterministic encoding,
    /// not a cryptographic digest or signature.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"luminous.symmetry_quotient.task_metadata.v1\0".to_vec();
        for field in self.fields() {
            append_string(&mut bytes, field);
        }
        bytes
    }
}

/// Canonical bytes for the exact V1 search problem plus its complete metadata.
pub fn canonical_complete_search_problem_bytes(
    problem: &SearchProblemV1,
    metadata: &SearchProblemMetadataV1,
) -> Vec<u8> {
    let mut bytes = b"luminous.symmetry_quotient.complete_search_problem.v1\0".to_vec();
    let core = canonical_search_problem_bytes(problem);
    let metadata_bytes = metadata.canonical_bytes();
    append_bytes(&mut bytes, &core);
    append_bytes(&mut bytes, &metadata_bytes);
    bytes
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompleteTaskQuotientCertificateV1 {
    pub schema_version: u32,
    /// Exact bytes for the V1 task plus all declared experiment/evaluator metadata.
    pub complete_problem_contract_bytes: Vec<u8>,
    pub core_certificate: QuotientCertificateV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompleteTaskCompileOutcomeV1 {
    Authorized(CompleteTaskQuotientCertificateV1),
    Blocked(TaskInvarianceViolationV1),
}

/// Compile only after every required task-contract identity is present.
/// The inner compiler still independently checks the finite group, action,
/// per-candidate semantics, complete orbit cover and stabilizer witnesses.
pub fn compile_complete_task_quotient(
    problem: &SearchProblemV1,
    metadata: &SearchProblemMetadataV1,
    action: &FiniteActionV1,
    budget: QualificationBudgetV1,
) -> Result<CompleteTaskCompileOutcomeV1, QuotientErrorV1> {
    metadata.validate()?;
    let complete_problem_contract_bytes =
        canonical_complete_search_problem_bytes(problem, metadata);
    match compile_quotient(problem, action, budget)? {
        CompileOutcomeV1::Authorized(core_certificate) => {
            Ok(CompleteTaskCompileOutcomeV1::Authorized(
                CompleteTaskQuotientCertificateV1 {
                    schema_version: 1,
                    complete_problem_contract_bytes,
                    core_certificate,
                },
            ))
        }
        CompileOutcomeV1::Blocked(violation) => {
            Ok(CompleteTaskCompileOutcomeV1::Blocked(violation))
        }
    }
}

/// Verify the complete metadata/task identity before replaying the core proof.
/// Changing an evaluator, units, chart, dataset, solver or stopping policy
/// invalidates the certificate even if the finite action remains invariant.
pub fn verify_complete_task_quotient(
    problem: &SearchProblemV1,
    metadata: &SearchProblemMetadataV1,
    action: &FiniteActionV1,
    certificate: &CompleteTaskQuotientCertificateV1,
    budget: QualificationBudgetV1,
) -> Result<VerificationReceiptV1, QuotientErrorV1> {
    metadata.validate()?;
    if certificate.schema_version != 1
        || certificate.complete_problem_contract_bytes
            != canonical_complete_search_problem_bytes(problem, metadata)
    {
        return Err(QuotientErrorV1::CertificateIdentityMismatch);
    }
    verify_certificate(problem, action, &certificate.core_certificate, budget)
}

/// Replay the complete contract and core proof before lifting every member of
/// a representative orbit into the original candidate universe.
pub fn lift_complete_task_representative_solution(
    problem: &SearchProblemV1,
    metadata: &SearchProblemMetadataV1,
    action: &FiniteActionV1,
    certificate: &CompleteTaskQuotientCertificateV1,
    representative_index: usize,
    budget: QualificationBudgetV1,
) -> Result<Vec<LiftedCandidateV1>, QuotientErrorV1> {
    metadata.validate()?;
    if certificate.schema_version != 1
        || certificate.complete_problem_contract_bytes
            != canonical_complete_search_problem_bytes(problem, metadata)
    {
        return Err(QuotientErrorV1::CertificateIdentityMismatch);
    }
    // The core lifting routine performs the one required full certificate
    // replay; avoid replaying the core twice under the same resource budget.
    lift_representative_solution(
        problem,
        action,
        &certificate.core_certificate,
        representative_index,
        budget,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CandidateSemanticsV1, FiniteGroupV1,
    };

    fn metadata() -> SearchProblemMetadataV1 {
        SearchProblemMetadataV1 {
            candidate_universe_revision: "candidate-universe-fixture-v1".to_owned(),
            action_domain_id: "finite-c2-domain-v1".to_owned(),
            objective_definition_id: "integer-minimise-cost-v1".to_owned(),
            constraint_schema_id: "one-integer-constraint-v1".to_owned(),
            unit_system_id: "dimensionless-integer-units-v1".to_owned(),
            coordinate_chart_id: "not-applicable".to_owned(),
            initial_condition_id: "not-applicable".to_owned(),
            boundary_condition_family_id: "closed-finite-domain-v1".to_owned(),
            observable_map_id: "candidate-identity-output-v1".to_owned(),
            dataset_revision: "synthetic-fixture-v1".to_owned(),
            evaluator_revision: "exact-evaluator-v1".to_owned(),
            source_revision: "fixture-source-revision-1".to_owned(),
            solver_profile_id: "exhaustive-enumeration-v1".to_owned(),
            precision_policy_id: "exact-i64-v1".to_owned(),
            resource_budget_policy_id: "qualification-budget-v1".to_owned(),
            termination_policy_id: "complete-enumeration-v1".to_owned(),
            randomness_policy_id: "not-applicable".to_owned(),
            intervention_family_id: "not-applicable".to_owned(),
            measurement_model_id: "exact-finite-candidate-observation-v1".to_owned(),
            noise_model_id: "zero-noise-exact-outcomes-v1".to_owned(),
            context_selection_policy_id: "single-declared-context-v1".to_owned(),
        }
    }

    fn fixture() -> (SearchProblemV1, FiniteActionV1) {
        let semantics = CandidateSemanticsV1 {
            in_search_domain: true,
            admissible: true,
            feasible: true,
            objective_cost: 5,
            constraint_signature: vec![1],
            boundary_condition_id: "boundary-a".to_owned(),
            output_equivalence_id: "output-a".to_owned(),
        };
        let problem = SearchProblemV1 {
            problem_id: "complete-task-contract-c2-v1".to_owned(),
            candidate_ids: vec!["candidate-a".to_owned(), "candidate-b".to_owned()],
            semantics: vec![semantics.clone(), semantics],
        };
        let action = FiniteActionV1 {
            action_id: "complete-task-swap-v1".to_owned(),
            group: FiniteGroupV1 {
                group_id: "C2-complete-task-v1".to_owned(),
                identity: 0,
                multiplication: vec![vec![0, 1], vec![1, 0]],
            },
            permutations: vec![vec![0, 1], vec![1, 0]],
        };
        (problem, action)
    }

    #[test]
    fn complete_contract_compiles_verifies_and_lifts() {
        let (problem, action) = fixture();
        let metadata = metadata();
        let outcome = compile_complete_task_quotient(
            &problem,
            &metadata,
            &action,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        let CompleteTaskCompileOutcomeV1::Authorized(certificate) = outcome else {
            panic!("identical task semantics must admit the C2 quotient");
        };
        let receipt = verify_complete_task_quotient(
            &problem,
            &metadata,
            &action,
            &certificate,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(receipt.orbit_count, 1);
        let lifted = lift_complete_task_representative_solution(
            &problem,
            &metadata,
            &action,
            &certificate,
            0,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(
            lifted.iter().map(|item| item.candidate_id.as_str()).collect::<Vec<_>>(),
            vec!["candidate-a", "candidate-b"]
        );
    }

    #[test]
    fn changing_provenance_or_intervention_metadata_invalidates_replay() {
        let (problem, action) = fixture();
        let original = metadata();
        assert_eq!(SearchProblemMetadataV1::FIELD_COUNT, 21);
        assert_eq!(original.fields().len(), SearchProblemMetadataV1::FIELD_COUNT);
        let outcome = compile_complete_task_quotient(
            &problem,
            &original,
            &action,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        let CompleteTaskCompileOutcomeV1::Authorized(certificate) = outcome else {
            panic!("fixture must authorize");
        };

        for mutate in 0..7 {
            let mut changed = original.clone();
            match mutate {
                0 => changed.dataset_revision = "dataset-revision-2".to_owned(),
                1 => changed.evaluator_revision = "evaluator-revision-2".to_owned(),
                2 => changed.solver_profile_id = "solver-profile-2".to_owned(),
                3 => changed.unit_system_id = "metres-instead-of-dimensionless".to_owned(),
                4 => changed.intervention_family_id = "intervention-family-2".to_owned(),
                5 => changed.measurement_model_id = "measurement-model-2".to_owned(),
                _ => changed.noise_model_id = "noise-model-2".to_owned(),
            }
            assert_eq!(
                verify_complete_task_quotient(
                    &problem,
                    &changed,
                    &action,
                    &certificate,
                    QualificationBudgetV1::default(),
                ),
                Err(QuotientErrorV1::CertificateIdentityMismatch)
            );
        }
    }

    #[test]
    fn unknown_or_missing_complete_metadata_fails_closed() {
        let (problem, action) = fixture();
        let mut unknown = metadata();
        unknown.evaluator_revision = "unknown".to_owned();
        assert_eq!(
            compile_complete_task_quotient(
                &problem,
                &unknown,
                &action,
                QualificationBudgetV1::default()
            ),
            Err(QuotientErrorV1::InvalidIdentifier)
        );
        let mut empty = metadata();
        empty.resource_budget_policy_id.clear();
        assert_eq!(
            compile_complete_task_quotient(
                &problem,
                &empty,
                &action,
                QualificationBudgetV1::default()
            ),
            Err(QuotientErrorV1::InvalidIdentifier)
        );
    }

    #[test]
    fn explicit_not_applicable_is_valid_and_bytes_are_deterministic() {
        let (problem, _) = fixture();
        let metadata = metadata();
        metadata.validate().unwrap();
        let a = canonical_complete_search_problem_bytes(&problem, &metadata);
        let b = canonical_complete_search_problem_bytes(&problem, &metadata);
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn task_semantic_violation_remains_blocked_under_complete_contract() {
        let (mut problem, action) = fixture();
        problem.semantics[1].boundary_condition_id = "different-boundary".to_owned();
        let outcome = compile_complete_task_quotient(
            &problem,
            &metadata(),
            &action,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert!(matches!(outcome, CompleteTaskCompileOutcomeV1::Blocked(_)));
    }
}
