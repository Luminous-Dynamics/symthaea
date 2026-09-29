//! Deterministic scientific investigation traces.
//!
//! This contract stitches together the evidence-plane lifecycle without
//! collapsing reasoning, experiment selection, execution, observation,
//! replication, or criterion qualification into one semantic event.
//!
//! A trace is provenance/orchestration metadata. It is not a truth score,
//! consensus mechanism, or criterion-completion assertion.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const VERSION: &str = "1.0.0";
const DOMAIN: &[u8] = b"symthaea:scientific-investigation-trace:v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvestigationStageKind {
    Hypothesis,
    ModelCouncilSelection,
    ProspectivePrediction,
    ExperimentSelection,
    ExternalExecution,
    Observation,
    IndependentAssessment,
    Replication,
    CriterionEvidence,
    ProvenanceProof,
}

impl InvestigationStageKind {
    fn ordinal(self) -> u8 {
        match self {
            Self::Hypothesis => 0,
            Self::ModelCouncilSelection => 1,
            Self::ProspectivePrediction => 2,
            Self::ExperimentSelection => 3,
            Self::ExternalExecution => 4,
            Self::Observation => 5,
            Self::IndependentAssessment => 6,
            Self::Replication => 7,
            Self::CriterionEvidence => 8,
            Self::ProvenanceProof => 9,
        }
    }

    fn computational(self) -> bool {
        matches!(
            self,
            Self::Hypothesis
                | Self::ModelCouncilSelection
                | Self::ProspectivePrediction
                | Self::ExperimentSelection
                | Self::ProvenanceProof
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvestigationOutcome {
    Pending,
    Supports,
    Contradicts,
    Null,
    Inconclusive,
    ProceduralInvalid,
    Partial,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvestigationArtifactRef {
    pub artifact_id: String,
    pub artifact_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvestigationStage {
    pub stage_id: String,
    pub kind: InvestigationStageKind,
    pub artifact: InvestigationArtifactRef,
    pub outcome: InvestigationOutcome,
    pub parent_stage_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificInvestigationTrace {
    pub trace_version: String,
    pub investigation_id: String,
    pub challenge_id: Option<String>,
    pub stages: Vec<InvestigationStage>,
    pub trace_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvestigationTraceError {
    InvalidTrace,
    InvalidDigest,
    DuplicateStage,
    NonCanonicalStageOrder,
    MissingParent,
    ParentAfterChild,
    InvalidProspectiveOrdering,
    ObservationWithoutExecution,
    ComputationalCriterionEvidence,
    ArtifactSubstitution,
    TraceDigestMismatch,
}

impl ScientificInvestigationTrace {
    pub fn new(
        investigation_id: impl Into<String>,
        challenge_id: Option<String>,
        stages: Vec<InvestigationStage>,
    ) -> Result<Self, InvestigationTraceError> {
        let trace = Self {
            trace_version: VERSION.into(),
            investigation_id: investigation_id.into(),
            challenge_id,
            stages,
            trace_digest: String::new(),
        };
        trace.validate()?;
        let mut trace = trace;
        trace.trace_digest = trace.compute_digest();
        Ok(trace)
    }

    pub fn verify(&self) -> Result<(), InvestigationTraceError> {
        if self.trace_version != VERSION
            || self.investigation_id.trim().is_empty()
            || !self.trace_digest.starts_with("sha256:")
        {
            return Err(InvestigationTraceError::InvalidTrace);
        }

        let mut seen = std::collections::BTreeSet::new();
        let mut positions = std::collections::BTreeMap::new();

        for (index, stage) in self.stages.iter().enumerate() {
            if stage.stage_id.trim().is_empty()
                || !seen.insert(stage.stage_id.clone())
                || !valid_digest(&stage.artifact.artifact_digest)
                || stage.artifact.artifact_id.trim().is_empty()
            {
                return Err(InvestigationTraceError::InvalidTrace);
            }
            positions.insert(stage.stage_id.clone(), index);
        }

        for window in self.stages.windows(2) {
            if window[0].kind.ordinal() > window[1].kind.ordinal() {
                return Err(InvestigationTraceError::NonCanonicalStageOrder);
            }
        }

        for stage in &self.stages {
            for parent in &stage.parent_stage_ids {
                let parent_pos = *positions
                    .get(parent)
                    .ok_or(InvestigationTraceError::MissingParent)?;
                let child_pos = positions[&stage.stage_id];
                if parent_pos >= child_pos {
                    return Err(InvestigationTraceError::ParentAfterChild);
                }
            }
        }

        self.validate_semantics()?;

        if self.trace_digest != self.compute_digest() {
            return Err(InvestigationTraceError::TraceDigestMismatch);
        }

        Ok(())
    }

    fn validate(&self) -> Result<(), InvestigationTraceError> {
        self.verify_without_digest()?;
        Ok(())
    }

    fn verify_without_digest(&self) -> Result<(), InvestigationTraceError> {
        if self.trace_version != VERSION || self.investigation_id.trim().is_empty() {
            return Err(InvestigationTraceError::InvalidTrace);
        }

        let mut seen = std::collections::BTreeSet::new();
        let mut positions = std::collections::BTreeMap::new();

        for (index, stage) in self.stages.iter().enumerate() {
            if stage.stage_id.trim().is_empty()
                || !seen.insert(stage.stage_id.clone())
                || !valid_digest(&stage.artifact.artifact_digest)
                || stage.artifact.artifact_id.trim().is_empty()
            {
                return Err(InvestigationTraceError::InvalidTrace);
            }
            positions.insert(stage.stage_id.clone(), index);
        }

        for window in self.stages.windows(2) {
            if window[0].kind.ordinal() > window[1].kind.ordinal() {
                return Err(InvestigationTraceError::NonCanonicalStageOrder);
            }
        }

        for stage in &self.stages {
            for parent in &stage.parent_stage_ids {
                let parent_pos = *positions
                    .get(parent)
                    .ok_or(InvestigationTraceError::MissingParent)?;
                if parent_pos >= positions[&stage.stage_id] {
                    return Err(InvestigationTraceError::ParentAfterChild);
                }
            }
        }

        self.validate_semantics()
    }

    fn validate_semantics(&self) -> Result<(), InvestigationTraceError> {
        let has = |kind: InvestigationStageKind| {
            self.stages.iter().any(|stage| stage.kind == kind)
        };

        if has(InvestigationStageKind::Observation) && !has(InvestigationStageKind::ExternalExecution) {
            return Err(InvestigationTraceError::ObservationWithoutExecution);
        }

        if let (Some(prediction), Some(observation)) = (
            self.stages.iter().position(|s| s.kind == InvestigationStageKind::ProspectivePrediction),
            self.stages.iter().position(|s| s.kind == InvestigationStageKind::Observation),
        ) {
            if prediction >= observation {
                return Err(InvestigationTraceError::InvalidProspectiveOrdering);
            }
        }

        if self.stages.iter().any(|stage| {
            stage.kind == InvestigationStageKind::CriterionEvidence && stage.kind.computational()
        }) {
            return Err(InvestigationTraceError::ComputationalCriterionEvidence);
        }

        Ok(())
    }

    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(DOMAIN);
        put(&mut h, &self.trace_version);
        put(&mut h, &self.investigation_id);
        put(&mut h, self.challenge_id.as_deref().unwrap_or(""));
        for stage in &self.stages {
            put(&mut h, &stage.stage_id);
            put(&mut h, &stage.kind.ordinal().to_string());
            put(&mut h, &stage.artifact.artifact_id);
            put(&mut h, &stage.artifact.artifact_digest);
            put(&mut h, &format!("{:?}", stage.outcome));
            for parent in &stage.parent_stage_ids {
                put(&mut h, parent);
            }
        }
        format!("sha256:{:x}", h.finalize())
    }

    pub fn artifact_digest_for(&self, stage_id: &str) -> Option<&str> {
        self.stages
            .iter()
            .find(|stage| stage.stage_id == stage_id)
            .map(|stage| stage.artifact.artifact_digest.as_str())
    }
}

fn valid_digest(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value.as_bytes()[7..].iter().all(|b| b.is_ascii_hexdigit())
}

fn put(h: &mut Sha256, value: &str) {
    h.update((value.len() as u64).to_be_bytes());
    h.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        format!("sha256:{}", ch.to_string().repeat(64))
    }

    fn stage(id: &str, kind: InvestigationStageKind, parent: &[&str]) -> InvestigationStage {
        InvestigationStage {
            stage_id: id.into(),
            kind,
            artifact: InvestigationArtifactRef {
                artifact_id: format!("artifact:{id}"),
                artifact_digest: digest('a'),
            },
            outcome: InvestigationOutcome::Pending,
            parent_stage_ids: parent.iter().map(|p| (*p).into()).collect(),
        }
    }

    #[test]
    fn prospective_prediction_must_precede_observation() {
        let stages = vec![
            stage("execution", InvestigationStageKind::ExternalExecution, &[]),
            stage("observation", InvestigationStageKind::Observation, &["execution"]),
            stage("prediction", InvestigationStageKind::ProspectivePrediction, &[]),
        ];
        assert_eq!(
            ScientificInvestigationTrace::new("investigation:1", None, stages),
            Err(InvestigationTraceError::NonCanonicalStageOrder)
        );
    }

    #[test]
    fn observation_requires_external_execution() {
        let stages = vec![stage("observation", InvestigationStageKind::Observation, &[])];
        assert_eq!(
            ScientificInvestigationTrace::new("investigation:1", None, stages),
            Err(InvestigationTraceError::ObservationWithoutExecution)
        );
    }

    #[test]
    fn missing_parent_is_rejected() {
        let stages = vec![stage("observation", InvestigationStageKind::Observation, &["missing"])];
        assert_eq!(
            ScientificInvestigationTrace::new("investigation:1", None, stages),
            Err(InvestigationTraceError::MissingParent)
        );
    }

    #[test]
    fn negative_outcomes_are_valid_trace_data() {
        let mut observation = stage("observation", InvestigationStageKind::Observation, &["execution"]);
        observation.outcome = InvestigationOutcome::Contradicts;
        let trace = ScientificInvestigationTrace::new(
            "investigation:1",
            Some("millennium-008".into()),
            vec![
                stage("execution", InvestigationStageKind::ExternalExecution, &[]),
                observation,
            ],
        )
        .unwrap();
        assert!(trace.verify().is_ok());
    }

    #[test]
    fn artifact_substitution_changes_trace_digest() {
        let base = ScientificInvestigationTrace::new(
            "investigation:1",
            None,
            vec![stage("hypothesis", InvestigationStageKind::Hypothesis, &[])],
        )
        .unwrap();
        let mut altered = base.clone();
        altered.stages[0].artifact.artifact_id = "artifact:substituted".into();
        assert_ne!(base.trace_digest, altered.compute_digest());
    }

    #[test]
    fn serde_roundtrip() {
        let trace = ScientificInvestigationTrace::new(
            "investigation:1",
            None,
            vec![stage("hypothesis", InvestigationStageKind::Hypothesis, &[])],
        )
        .unwrap();
        let encoded = serde_json::to_vec(&trace).unwrap();
        assert_eq!(
            serde_json::from_slice::<ScientificInvestigationTrace>(&encoded).unwrap(),
            trace
        );
    }
}
