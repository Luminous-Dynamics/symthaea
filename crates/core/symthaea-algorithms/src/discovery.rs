//! Zero-authority discovery protocol.
//!
//! This module describes search runs and candidate artifacts. It deliberately has no process,
//! filesystem-mutation, Git, merge, activation, or promotion API.

use crate::evaluation::EvaluationReceipt;
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
    #[error("search budget values must be positive")]
    ZeroBudget,
    #[error("automated discovery policy does not admit risk class {0:?}")]
    RiskNotAdmitted(DiscoveryRisk),
    #[error("candidate problem does not match the discovery run")]
    ProblemMismatch,
    #[error("candidate lineage does not match the candidate implementation")]
    LineageMismatch,
    #[error("evaluation receipt does not match the candidate implementation/problem")]
    EvaluationMismatch,
    #[error("candidate implementation is already archived")]
    DuplicateCandidate,
    #[error("discovery run identity does not match its canonical fields")]
    IdentityMismatch,
    #[error("candidate source/artifact reference must not be empty")]
    EmptyArtifactReference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
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
}

/// Maximum risk class an automated discovery run may accept.
///
/// The default admits only ordinary computational optimization. Security-sensitive and
/// safety-critical discovery require a separate explicit policy decision and still gain no
/// promotion/runtime authority from this type.
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
        if implementation.problem_id != run.problem_id {
            return Err(DiscoveryError::ProblemMismatch);
        }
        lineage
            .validate_for(&implementation)
            .map_err(|_| DiscoveryError::LineageMismatch)?;
        Ok(Self {
            run_id: run.id.clone(),
            implementation,
            lineage,
            artifact,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchivedCandidate {
    pub proposal: CandidateProposal,
    pub evaluation: Option<EvaluationReceipt>,
}

/// In-memory/value-level candidate archive.
///
/// Persistence is intentionally left to a caller-owned adapter. The archive cannot write files,
/// invoke Git, or mint production authority.
#[derive(Debug, Default)]
pub struct CandidateArchive {
    candidates: BTreeMap<ImplementationId, ArchivedCandidate>,
}

impl CandidateArchive {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, proposal: CandidateProposal) -> Result<(), DiscoveryError> {
        let key = proposal.implementation.id.clone();
        if self.candidates.contains_key(&key) {
            return Err(DiscoveryError::DuplicateCandidate);
        }
        self.candidates.insert(
            key,
            ArchivedCandidate {
                proposal,
                evaluation: None,
            },
        );
        Ok(())
    }

    pub fn attach_evaluation(
        &mut self,
        implementation_id: &ImplementationId,
        receipt: EvaluationReceipt,
    ) -> Result<(), DiscoveryError> {
        let Some(candidate) = self.candidates.get_mut(implementation_id) else {
            return Err(DiscoveryError::EvaluationMismatch);
        };
        if receipt.implementation_id != candidate.proposal.implementation.id
            || receipt.problem_id != candidate.proposal.implementation.problem_id
        {
            return Err(DiscoveryError::EvaluationMismatch);
        }
        candidate.evaluation = Some(receipt);
        Ok(())
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
            .filter_map(|candidate| candidate.evaluation.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let implementation = ImplementationRecord::new(
            run.problem_id.clone(),
            AlgorithmId(cid("algorithm", "family")),
            "candidate://run/1",
            cid("artifact", "candidate-1"),
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
                cid("patch", "candidate-1"),
                "archive://candidate-1.patch",
            )
            .unwrap(),
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
    fn zero_budget_fails_closed() {
        assert_eq!(
            SearchBudget::new(0, 1, 1).unwrap_err(),
            DiscoveryError::ZeroBudget
        );
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
    fn archive_rejects_duplicate_candidates() {
        let problem = problem(DiscoveryRisk::Ordinary);
        let run = run(&problem);
        let proposal = proposal(&run);
        let mut archive = CandidateArchive::new();
        archive.insert(proposal.clone()).unwrap();
        assert_eq!(
            archive.insert(proposal).unwrap_err(),
            DiscoveryError::DuplicateCandidate
        );
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
        let implementation = ImplementationRecord::new(
            second.id,
            AlgorithmId(cid("algorithm", "other")),
            "candidate://other",
            cid("artifact", "other"),
            None,
        )
        .unwrap();
        let lineage = AlgorithmLineage::new(implementation.id.clone(), vec![], vec![]).unwrap();
        let artifact = CandidateArtifact::new(
            CandidateArtifactKind::SourceTree,
            cid("artifact", "tree"),
            "archive://tree",
        )
        .unwrap();
        assert_eq!(
            CandidateProposal::new(&run, implementation, lineage, artifact).unwrap_err(),
            DiscoveryError::ProblemMismatch
        );
    }
}
