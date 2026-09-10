// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Semantic qualification of persisted generator-local proposal evidence.
//!
//! This layer composes raw proposal/trace coverage, exact semantic proposal binding, and the
//! policy-learning qualification gate. It also closes the no-op-only baseline edge: every recorded
//! attempt must bind the exact supplied baseline implementation artifact even when no concrete
//! transformation trial exists.

use crate::proposal_coverage::{
    validate_forge_raw_proposal_coverage, ForgeProposalCoverageError,
};
use crate::proposal_exposure::{
    ForgeProposalExposureArchive, ForgeProposalExposureError, ForgeProposalLearningQualification,
    ForgeProposalPolicy,
};
use crate::proposal_recording::{exposure_from_raw_record, ForgeProposalRecordingError};
use crate::proposal_trace::{ForgeRawProposalArchive, ForgeRawProposalError};
use crate::trace::ForgeTraceEvent;
use serde::Serialize;
use symthaea_algorithms::discovery::{DiscoveryError, DiscoveryRun};
use symthaea_algorithms::observation::ObservationStore;
use symthaea_algorithms::{ContentId, ImplementationRecord, RegistryError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalQualificationError {
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    #[error(transparent)]
    Registry(#[from] RegistryError),
    #[error(transparent)]
    Raw(#[from] ForgeRawProposalError),
    #[error(transparent)]
    Coverage(#[from] ForgeProposalCoverageError),
    #[error(transparent)]
    Recording(#[from] ForgeProposalRecordingError),
    #[error(transparent)]
    Exposure(#[from] ForgeProposalExposureError),
    #[error("raw proposal attempt baseline does not match the supplied baseline implementation")]
    BaselineArtifactMismatch,
    #[error("qualified proposal-evidence identity does not match canonical fields")]
    IdentityMismatch,
}

/// Exact semantic bridge from one generator-local raw archive to the proposal-learning gate.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeQualifiedProposalEvidence {
    id: ContentId,
    run_id: ContentId,
    baseline_implementation_id: ContentId,
    raw_archive_id: ContentId,
    semantic_archive: ForgeProposalExposureArchive,
    qualification: ForgeProposalLearningQualification,
}

impl ForgeQualifiedProposalEvidence {
    #[allow(clippy::too_many_arguments)]
    pub fn from_raw(
        run: &DiscoveryRun,
        baseline: &ImplementationRecord,
        trace: &[ForgeTraceEvent],
        observations: &ObservationStore,
        policy: &ForgeProposalPolicy,
        raw: &ForgeRawProposalArchive,
    ) -> Result<Self, ForgeProposalQualificationError> {
        run.validate()?;
        baseline.validate()?;
        raw.validate()?;
        validate_forge_raw_proposal_coverage(trace, observations, raw)?;

        // This check is intentionally independent of concrete transformation trials. A run whose
        // every proposal was a no-op still has to prove it searched from the declared baseline.
        if raw.records().iter().any(|record| {
            record.attempt_id().baseline_artifact_id() != &baseline.artifact_id
        }) {
            return Err(ForgeProposalQualificationError::BaselineArtifactMismatch);
        }

        let exposures = raw
            .records()
            .iter()
            .map(|record| exposure_from_raw_record(run, policy, record))
            .collect::<Result<Vec<_>, _>>()?;
        let semantic_archive = ForgeProposalExposureArchive::new(run, policy, exposures)?;
        let qualification = ForgeProposalLearningQualification::qualify(
            run,
            baseline,
            trace,
            observations,
            policy,
            &semantic_archive,
        )?;

        let baseline_implementation_id = baseline.id.as_content_id().clone();
        let id = derive_id(
            &run.id,
            &baseline_implementation_id,
            raw.id(),
            semantic_archive.id(),
            qualification.id(),
        );
        let result = Self {
            id,
            run_id: run.id.clone(),
            baseline_implementation_id,
            raw_archive_id: raw.id().clone(),
            semantic_archive,
            qualification,
        };
        result.validate_identity()?;
        Ok(result)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn run_id(&self) -> &ContentId { &self.run_id }
    pub fn baseline_implementation_id(&self) -> &ContentId { &self.baseline_implementation_id }
    pub fn raw_archive_id(&self) -> &ContentId { &self.raw_archive_id }
    pub fn semantic_archive(&self) -> &ForgeProposalExposureArchive { &self.semantic_archive }
    pub fn qualification(&self) -> &ForgeProposalLearningQualification { &self.qualification }

    pub fn validate_identity(&self) -> Result<(), ForgeProposalQualificationError> {
        self.qualification.validate_identity()?;
        let expected = derive_id(
            &self.run_id,
            &self.baseline_implementation_id,
            &self.raw_archive_id,
            self.semantic_archive.id(),
            self.qualification.id(),
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalQualificationError::IdentityMismatch)
        }
    }
}

fn derive_id(
    run_id: &ContentId,
    baseline_implementation_id: &ContentId,
    raw_archive_id: &ContentId,
    semantic_archive_id: &ContentId,
    qualification_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-qualified-proposal-evidence.v1",
        [
            run_id.as_str().as_bytes(),
            baseline_implementation_id.as_str().as_bytes(),
            raw_archive_id.as_str().as_bytes(),
            semantic_archive_id.as_str().as_bytes(),
            qualification_id.as_str().as_bytes(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::full_source_artifact_id;
    use crate::mutations::{ComparisonOperatorSwap, Mutator};
    use crate::observations as forge_observations;
    use crate::proposal_recording::proposal_policy_for_mutator;
    use crate::proposal_trace::ForgeRawProposalRecord;
    use crate::trace::{ForgeAttemptId, ForgeTraceEvent};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use symthaea_algorithms::discovery::{DiscoveryPolicy, SearchBudget};
    use symthaea_algorithms::observation::ObservationStore;
    use symthaea_algorithms::{
        AlgorithmProvenance, AlgorithmRecord, DeterminismRequirement, DiscoveryRisk,
        ProblemSpec, SemanticGuarantee,
    };

    fn semantic_fixture(
        baseline_artifact: ContentId,
    ) -> (DiscoveryRun, ImplementationRecord, ForgeProposalPolicy) {
        let problem = ProblemSpec::new(
            "proposal-qualification-test",
            "Exact fixture.",
            SemanticGuarantee::Exact,
            DeterminismRequirement::Required,
            vec!["matches oracle".into()],
            DiscoveryRisk::Ordinary,
        )
        .unwrap();
        let algorithm = AlgorithmRecord::new(
            problem.id.clone(),
            "baseline",
            "fixture baseline",
            AlgorithmProvenance::HumanAuthored,
        )
        .unwrap();
        let baseline = ImplementationRecord::new(
            problem.id.clone(),
            algorithm.id,
            "repo://src/f.rs",
            baseline_artifact,
            None,
        )
        .unwrap();
        let run = DiscoveryRun::new(
            &problem,
            DiscoveryPolicy::default(),
            ContentId::derive("generator", [b"forge-proposal-qualified-v1".as_slice()]),
            "abc123",
            SearchBudget::new(8, 2, 8).unwrap(),
            7,
        )
        .unwrap();
        let policy = proposal_policy_for_mutator(&run, &Mutator::new(vec![
            Box::new(ComparisonOperatorSwap),
        ]))
        .unwrap();
        (run, baseline, policy)
    }

    fn no_op_history(
        attempt_baseline: ContentId,
        run: &DiscoveryRun,
    ) -> (Vec<ForgeTraceEvent>, ObservationStore, ForgeRawProposalArchive) {
        let attempt = ForgeAttemptId::derive(&attempt_baseline, run.seed, 0, 0);
        let mut body: syn::Block = syn::parse_str("{ \"no comparison\" }").unwrap();
        let mut rng = StdRng::seed_from_u64(11);
        let mutator = Mutator::new(vec![Box::new(ComparisonOperatorSwap)]);
        let recorded = mutator.mutate_one_recorded(&mut body, &mut rng);
        let raw_record = ForgeRawProposalRecord::from_recorded(
            attempt.clone(),
            attempt_baseline.clone(),
            &recorded,
        )
        .unwrap();
        let no_candidate = forge_observations::no_candidate(
            &attempt,
            "no-eligible-ast-mutation",
            &attempt_baseline,
        )
        .unwrap();
        let summary = forge_observations::search_summary(
            1, 1, 0, 0, 0, 0, 0, None, None,
        )
        .unwrap();
        let trace = vec![
            ForgeTraceEvent::no_candidate(attempt, 0, no_candidate.id().clone()),
            ForgeTraceEvent::completed(summary.id().clone()),
        ];
        let observations = ObservationStore::from_objects(vec![no_candidate, summary]).unwrap();
        let raw = ForgeRawProposalArchive::from_records(vec![raw_record]).unwrap();
        (trace, observations, raw)
    }

    #[test]
    fn no_op_only_history_still_binds_exact_baseline() {
        let artifact = full_source_artifact_id("fn f() -> bool { true }\n");
        let (run, baseline, policy) = semantic_fixture(artifact.clone());
        let (trace, observations, raw) = no_op_history(artifact, &run);
        let qualified = ForgeQualifiedProposalEvidence::from_raw(
            &run,
            &baseline,
            &trace,
            &observations,
            &policy,
            &raw,
        )
        .unwrap();
        assert_eq!(qualified.raw_archive_id(), raw.id());
        assert_eq!(qualified.semantic_archive().exposures().len(), 1);
    }

    #[test]
    fn no_op_only_wrong_attempt_baseline_fails_closed() {
        let declared = full_source_artifact_id("fn f() -> bool { true }\n");
        let actual = full_source_artifact_id("fn f() -> bool { false }\n");
        let (run, baseline, policy) = semantic_fixture(declared);
        let (trace, observations, raw) = no_op_history(actual, &run);
        assert!(matches!(
            ForgeQualifiedProposalEvidence::from_raw(
                &run,
                &baseline,
                &trace,
                &observations,
                &policy,
                &raw,
            ),
            Err(ForgeProposalQualificationError::BaselineArtifactMismatch)
        ));
    }
}
