//! Zero-authority discovery protocol.
//!
//! This module describes search runs and candidate artifacts. It deliberately has no process,
//! filesystem-mutation, Git, merge, activation, or promotion API.

use crate::evaluation::{EvaluationError, EvaluationReceipt};
use crate::{
    AlgorithmLineage, ContentId, DiscoveryRisk, ImplementationId, ImplementationRecord, ProblemId,
    ProblemSpec, RegistryError,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum DiscoveryError {
    #[error(transparent)]
    Registry(#[from] RegistryError),
    #[error("evaluation receipt is invalid: {0}")]
    Evaluation(#[from] EvaluationError),
    #[error("search budget values must be positive")]
    ZeroBudget,
    #[error("automated discovery policy does not admit risk class {0:?}")]
    RiskNotAdmitted(DiscoveryRisk),
    #[error("candidate problem does not match the discovery run/archive")]
    ProblemMismatch,
    #[error("candidate run does not match this archive")]
    RunMismatch,
    #[error("candidate lineage does not match the candidate implementation")]
    LineageMismatch,
    #[error("candidate artifact content identity does not match the implementation artifact")]
    ArtifactMismatch,
    #[error("evaluation receipt does not match the candidate implementation/problem")]
    EvaluationMismatch,
    #[error("candidate implementation is already archived")]
    DuplicateCandidate,
    #[error("evaluation receipt is already archived for this candidate")]
    DuplicateEvaluation,
    #[error("discovery run identity does not match its canonical fields")]
    IdentityMismatch,
    #[error("candidate source/artifact reference must not be empty")]
    EmptyArtifactReference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchBudget {
    pub max_candidates: u64,
    pub max_generations: u64,
    pub max_evaluations: u64,
}

impl SearchBudget {
    pub fn new(
        max_candidates: u64,
        max_generations: u64,
        max_evaluations: u64,
    ) -> Result<Self, DiscoveryError> {
        if max_candidates == 0 || max_generations == 0 || max_evaluations == 0 {
            return Err(DiscoveryError::ZeroBudget);
        }
        Ok(Self {
            max_candidates,
            max_generations,
            max_evaluations,
        })
    }

    pub fn validate(&self) -> Result<(), DiscoveryError> {
        Self::new(
            self.max_candidates,
            self.max_generations,
            self.max_evaluations,
        )?;
        Ok(())
    }
}

/// Maximum risk class an automated discovery run may accept.
///
/// The default admits only ordinary computational optimization. Raising this ceiling is an
/// explicit research-policy decision and still grants no promotion/runtime authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryPolicy {
    pub maximum_risk: DiscoveryRisk,
}

impl Default for DiscoveryPolicy {
    fn default() -> Self {
        Self {
            maximum_risk: DiscoveryRisk::Ordinary,
        }
    }
}

impl DiscoveryPolicy {
    pub fn admits(&self, problem: &ProblemSpec) -> Result<(), DiscoveryError> {
        problem.validate()?;
        if problem.risk <= self.maximum_risk {
            Ok(())
        } else {
            Err(DiscoveryError::RiskNotAdmitted(problem.risk))
        }
    }
}

/// Immutable identity of one bounded search experiment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryRun {
    pub id: ContentId,
    pub problem_id: ProblemId,
    pub generator_id: ContentId,
    pub baseline_revision: String,
    pub budget: SearchBudget,
    pub seed: u64,
}

impl DiscoveryRun {
    pub fn new(
        problem: &ProblemSpec,
        policy: DiscoveryPolicy,
        generator_id: ContentId,
        baseline_revision: impl Into<String>,
        budget: SearchBudget,
        seed: u64,
    ) -> Result<Self, DiscoveryError> {
        policy.admits(problem)?;
        budget.validate()?;
        let baseline_revision = baseline_revision.into();
        if baseline_revision.trim().is_empty() {
            return Err(DiscoveryError::EmptyArtifactReference);
        }
        let id = Self::derive_id(
            &problem.id,
            &generator_id,
            &baseline_revision,
            budget,
            seed,
        );
        Ok(Self {
            id,
            problem_id: problem.id.clone(),
            generator_id,
            baseline_revision,
            budget,
            seed,
        })
    }

    pub fn validate(&self) -> Result<(), DiscoveryError> {
        self.budget.validate()?;
        if self.baseline_revision.trim().is_empty() {
            return Err(DiscoveryError::EmptyArtifactReference);
        }
        let expected = Self::derive_id(
            &self.problem_id,
            &self.generator_id,
            &self.baseline_revision,
            self.budget,
            self.seed,
        );
        if self.id == expected {
            Ok(())
        } else {
            Err(DiscoveryError::IdentityMismatch)
        }
    }

    fn derive_id(
        problem_id: &ProblemId,
        generator_id: &ContentId,
        baseline_revision: &str,
        budget: SearchBudget,
        seed: u64,
    ) -> ContentId {
        let candidates = budget.max_candidates.to_be_bytes();
        let generations = budget.max_generations.to_be_bytes();
        let evaluations = budget.max_evaluations.to_be_bytes();
        let seed = seed.to_be_bytes();
        ContentId::derive(
            "symthaea.discovery-run.v1",
            [
                problem_id.as_content_id().as_str().as_bytes(),
                generator_id.as_str().as_bytes(),
                baseline_revision.as_bytes(),
                candidates.as_slice(),
                generations.as_slice(),
                evaluations.as_slice(),
                seed.as_slice(),
            ],
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CandidateArtifactKind {
    UnifiedDiff,
    SourceTree,
    Expression,
    Configuration,
    Other,
}

/// Reference to candidate bytes stored outside this authority-free contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateArtifact {
    pub kind: CandidateArtifactKind,
    pub content_id: ContentId,
    pub reference: String,
}

impl CandidateArtifact {
    pub fn new(
        kind: CandidateArtifactKind,
        content_id: ContentId,
        reference: impl Into<String>,
    ) -> Result<Self, DiscoveryError> {
        let reference = reference.into();
        if reference.trim().is_empty() {
            return Err(DiscoveryError::EmptyArtifactReference);
        }
        Ok(Self {
            kind,
            content_id,
            reference,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateProposal {
    pub run_id: ContentId,
    pub implementation: ImplementationRecord,
    pub lineage: AlgorithmLineage,
    pub artifact: CandidateArtifact,
}

impl CandidateProposal {
    pub fn new(
        run: &DiscoveryRun,
        implementation: ImplementationRecord,
        lineage: AlgorithmLineage,
        artifact: CandidateArtifact,
    ) -> Result<Self, DiscoveryError> {
        run.validate()?;
        implementation.validate()?;
        if implementation.problem_id != run.problem_id {
            return Err(DiscoveryError::ProblemMismatch);
        }
        lineage
            .validate_for(&implementation)
            .map_err(|_| DiscoveryError::LineageMismatch)?;
        if artifact.content_id != implementation.artifact_id {
            return Err(DiscoveryError::ArtifactMismatch);
        }
        Ok(Self {
            run_id: run.id.clone(),
            implementation,
            lineage,
            artifact,
        })
    }
}

/// One candidate plus its append-only set of historical evaluation receipts.
///
/// The map key is the receipt content identity, so attaching a receipt never overwrites a prior
/// measurement. The fields are private to prevent callers from bypassing archive invariants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchivedCandidate {
    proposal: CandidateProposal,
    evaluations: BTreeMap<ContentId, EvaluationReceipt>,
}

impl ArchivedCandidate {
    pub fn proposal(&self) -> &CandidateProposal {
        &self.proposal
    }

    pub fn evaluations(&self) -> impl Iterator<Item = &EvaluationReceipt> {
        self.evaluations.values()
    }

    pub fn evaluation_count(&self) -> usize {
        self.evaluations.len()
    }
}

/// Archive scoped to exactly one discovery run.
///
/// Persistence is caller-owned. This type cannot write files, invoke Git, or mint production
/// authority.
#[derive(Debug)]
pub struct CandidateArchive {
    run_id: ContentId,
    problem_id: ProblemId,
    candidates: BTreeMap<ImplementationId, ArchivedCandidate>,
}

impl CandidateArchive {
    pub fn for_run(run: &DiscoveryRun) -> Result<Self, DiscoveryError> {
        run.validate()?;
        Ok(Self {
            run_id: run.id.clone(),
            problem_id: run.problem_id.clone(),
            candidates: BTreeMap::new(),
        })
    }

    pub fn insert(&mut self, proposal: CandidateProposal) -> Result<(), DiscoveryError> {
        if proposal.run_id != self.run_id {
            return Err(DiscoveryError::RunMismatch);
        }
        if proposal.implementation.problem_id != self.problem_id {
            return Err(DiscoveryError::ProblemMismatch);
        }
        proposal.implementation.validate()?;
        proposal
            .lineage
            .validate_for(&proposal.implementation)
            .map_err(|_| DiscoveryError::LineageMismatch)?;
        if proposal.artifact.content_id != proposal.implementation.artifact_id {
            return Err(DiscoveryError::ArtifactMismatch);
        }
        let key = proposal.implementation.id.clone();
        if self.candidates.contains_key(&key) {
            return Err(DiscoveryError::DuplicateCandidate);
        }
        self.candidates.insert(
            key,
            ArchivedCandidate {
                proposal,
                evaluations: BTreeMap::new(),
            },
        );
        Ok(())
    }

    pub fn attach_evaluation(
        &mut self,
        implementation_id: &ImplementationId,
        receipt: EvaluationReceipt,
    ) -> Result<(), DiscoveryError> {
        receipt.validate()?;
        let Some(candidate) = self.candidates.get_mut(implementation_id) else {
            return Err(DiscoveryError::EvaluationMismatch);
        };
        if receipt.implementation_id != candidate.proposal.implementation.id
            || receipt.problem_id != self.problem_id
        {
            return Err(DiscoveryError::EvaluationMismatch);
        }
        if candidate.evaluations.contains_key(&receipt.id) {
            return Err(DiscoveryError::DuplicateEvaluation);
        }
        candidate.evaluations.insert(receipt.id.clone(), receipt);
        Ok(())
    }

    pub fn run_id(&self) -> &ContentId {
        &self.run_id
    }

    pub fn get(&self, id: &ImplementationId) -> Option<&ArchivedCandidate> {
        self.candidates.get(id)
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    pub fn evaluated(&self) -> impl Iterator<Item = &EvaluationReceipt> {
        self.candidates
            .values()
            .flat_map(ArchivedCandidate::evaluations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::{
        CorrectnessVerdict, EvaluationContext, ObjectiveDirection, ObjectiveMeasurement,
    };
    use crate::{
        AlgorithmId, AlgorithmLineage, ContentId, DeterminismRequirement, DiscoveryRisk,
        ImplementationRecord, SemanticGuarantee,
    };

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    fn problem(risk: DiscoveryRisk) -> ProblemSpec {
        ProblemSpec::new(
            "test-problem",
            "Return the exact reference result.",
            SemanticGuarantee::Exact,
            DeterminismRequirement::Required,
            vec!["output equals oracle".into()],
            risk,
        )
        .unwrap()
    }

    fn run(problem: &ProblemSpec) -> DiscoveryRun {
        DiscoveryRun::new(
            problem,
            DiscoveryPolicy::default(),
            cid("generator", "deterministic-sweep-v1"),
            "abc123",
            SearchBudget::new(100, 10, 100).unwrap(),
            42,
        )
        .unwrap()
    }

    fn proposal(run: &DiscoveryRun) -> CandidateProposal {
        let artifact_id = cid("artifact", "candidate-1");
        let implementation = ImplementationRecord::new(
            run.problem_id.clone(),
            AlgorithmId(cid("algorithm", "family")),
            "candidate://run/1",
            artifact_id.clone(),
            None,
        )
        .unwrap();
        let lineage = AlgorithmLineage::new(implementation.id.clone(), vec![], vec![]).unwrap();
        CandidateProposal::new(
            run,
            implementation,
            lineage,
            CandidateArtifact::new(
                CandidateArtifactKind::UnifiedDiff,
                artifact_id,
                "archive://candidate-1.patch",
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn evaluation(proposal: &CandidateProposal, run_id: &str, latency: f64) -> EvaluationReceipt {
        EvaluationReceipt::new(
            proposal.implementation.problem_id.clone(),
            proposal.implementation.id.clone(),
            EvaluationContext::new(
                cid("evaluator", "criterion"),
                cid("oracle", "reference"),
                cid("inputs", "seeded"),
                cid("environment", "test-machine"),
                "abc123",
                "rust-1.96.0",
                "x86_64-test",
                vec![1, 2, 3],
            )
            .unwrap(),
            CorrectnessVerdict::Passed,
            cid("correctness", "pass"),
            vec![
                ObjectiveMeasurement::new(
                    "latency",
                    ObjectiveDirection::Minimize,
                    latency,
                    "ns/op",
                )
                .unwrap(),
            ],
            Some(run_id.into()),
        )
        .unwrap()
    }

    #[test]
    fn default_policy_rejects_security_sensitive_search() {
        let problem = problem(DiscoveryRisk::SecuritySensitive);
        assert_eq!(
            DiscoveryRun::new(
                &problem,
                DiscoveryPolicy::default(),
                cid("generator", "g"),
                "abc123",
                SearchBudget::new(1, 1, 1).unwrap(),
                1,
            )
            .unwrap_err(),
            DiscoveryError::RiskNotAdmitted(DiscoveryRisk::SecuritySensitive)
        );
    }

    #[test]
    fn zero_budget_fails_closed_even_after_deserialization_style_mutation() {
        assert_eq!(SearchBudget::new(0, 1, 1).unwrap_err(), DiscoveryError::ZeroBudget);
        let invalid = SearchBudget {
            max_candidates: 0,
            max_generations: 1,
            max_evaluations: 1,
        };
        assert_eq!(invalid.validate().unwrap_err(), DiscoveryError::ZeroBudget);
    }

    #[test]
    fn run_identity_changes_with_baseline_revision() {
        let problem = problem(DiscoveryRisk::Ordinary);
        let a = run(&problem);
        let b = DiscoveryRun::new(
            &problem,
            DiscoveryPolicy::default(),
            a.generator_id.clone(),
            "def456",
            a.budget,
            a.seed,
        )
        .unwrap();
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn candidate_artifact_must_match_implementation_artifact() {
        let problem = problem(DiscoveryRisk::Ordinary);
        let run = run(&problem);
        let implementation = ImplementationRecord::new(
            run.problem_id.clone(),
            AlgorithmId(cid("algorithm", "family")),
            "candidate://run/1",
            cid("artifact", "implementation"),
            None,
        )
        .unwrap();
        let lineage = AlgorithmLineage::new(implementation.id.clone(), vec![], vec![]).unwrap();
        let artifact = CandidateArtifact::new(
            CandidateArtifactKind::UnifiedDiff,
            cid("artifact", "different"),
            "archive://candidate.patch",
        )
        .unwrap();
        assert_eq!(
            CandidateProposal::new(&run, implementation, lineage, artifact).unwrap_err(),
            DiscoveryError::ArtifactMismatch
        );
    }

    #[test]
    fn archive_is_scoped_to_one_run_and_rejects_duplicates() {
        let problem = problem(DiscoveryRisk::Ordinary);
        let run = run(&problem);
        let proposal = proposal(&run);
        let mut archive = CandidateArchive::for_run(&run).unwrap();
        archive.insert(proposal.clone()).unwrap();
        assert_eq!(archive.run_id(), &run.id);
        assert_eq!(
            archive.insert(proposal).unwrap_err(),
            DiscoveryError::DuplicateCandidate
        );
    }

    #[test]
    fn archive_retains_multiple_evaluations_without_overwrite() {
        let problem = problem(DiscoveryRisk::Ordinary);
        let run = run(&problem);
        let proposal = proposal(&run);
        let implementation_id = proposal.implementation.id.clone();
        let first = evaluation(&proposal, "measurement-a", 10.0);
        let second = evaluation(&proposal, "measurement-b", 11.0);
        let mut archive = CandidateArchive::for_run(&run).unwrap();
        archive.insert(proposal).unwrap();
        archive.attach_evaluation(&implementation_id, first).unwrap();
        archive.attach_evaluation(&implementation_id, second).unwrap();
        assert_eq!(archive.get(&implementation_id).unwrap().evaluation_count(), 2);
        assert_eq!(archive.evaluated().count(), 2);
    }

    #[test]
    fn duplicate_evaluation_receipt_is_rejected_not_replaced() {
        let problem = problem(DiscoveryRisk::Ordinary);
        let run = run(&problem);
        let proposal = proposal(&run);
        let implementation_id = proposal.implementation.id.clone();
        let receipt = evaluation(&proposal, "measurement-a", 10.0);
        let mut archive = CandidateArchive::for_run(&run).unwrap();
        archive.insert(proposal).unwrap();
        archive
            .attach_evaluation(&implementation_id, receipt.clone())
            .unwrap();
        assert_eq!(
            archive.attach_evaluation(&implementation_id, receipt).unwrap_err(),
            DiscoveryError::DuplicateEvaluation
        );
        assert_eq!(archive.get(&implementation_id).unwrap().evaluation_count(), 1);
    }

    #[test]
    fn proposal_must_match_run_problem() {
        let first = problem(DiscoveryRisk::Ordinary);
        let run = run(&first);
        let second = ProblemSpec::new(
            "other-problem",
            "Return another exact result.",
            SemanticGuarantee::Exact,
            DeterminismRequirement::Required,
            vec![],
            DiscoveryRisk::Ordinary,
        )
        .unwrap();
        let artifact_id = cid("artifact", "other");
        let implementation = ImplementationRecord::new(
            second.id,
            AlgorithmId(cid("algorithm", "other")),
            "candidate://other",
            artifact_id.clone(),
            None,
        )
        .unwrap();
        let lineage = AlgorithmLineage::new(implementation.id.clone(), vec![], vec![]).unwrap();
        let artifact = CandidateArtifact::new(
            CandidateArtifactKind::SourceTree,
            artifact_id,
            "archive://tree",
        )
        .unwrap();
        assert_eq!(
            CandidateProposal::new(&run, implementation, lineage, artifact).unwrap_err(),
            DiscoveryError::ProblemMismatch
        );
    }
}
