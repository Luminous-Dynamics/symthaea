//! Planning-to-commitment bridge for scientific investigations.
//!
//! This receipt binds planning outputs to a frozen prospective prediction
//! without allowing planning or later recompilation to rewrite the prediction.
//! It is orchestration/provenance metadata, not evidence or a truth judgment.

use crate::scientific_investigation_trace::{
    InvestigationStageKind, ScientificInvestigationTrace,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const VERSION: &str = "1.0.0";
const DOMAIN: &[u8] = b"symthaea:investigation-planning-bridge:v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvestigationPlanningBridgeReceipt {
    pub receipt_version: String,
    pub investigation_id: String,
    pub planning_trace_digest: String,
    pub model_council_artifact_digest: String,
    pub compiled_prediction_artifact_digest: String,
    pub prospective_commitment_digest: String,
    pub experiment_selection_artifact_digest: String,
    pub prediction_stage_id: String,
    pub experiment_selection_stage_id: String,
    pub execution_stage_id: Option<String>,
    pub observation_stage_id: Option<String>,
    pub receipt_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvestigationPlanningBridgeError {
    InvalidReceipt,
    InvalidDigest,
    InvestigationMismatch,
    PlanningTraceMismatch,
    StageMissing,
    StageKindMismatch,
    CommitmentMismatch,
    CompilerSubstitution,
    ExecutionBeforeCommitment,
    ObservationBeforeExecution,
    ReceiptDigestMismatch,
}

impl InvestigationPlanningBridgeReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        investigation_id: impl Into<String>,
        planning_trace_digest: impl Into<String>,
        model_council_artifact_digest: impl Into<String>,
        compiled_prediction_artifact_digest: impl Into<String>,
        prospective_commitment_digest: impl Into<String>,
        experiment_selection_artifact_digest: impl Into<String>,
        prediction_stage_id: impl Into<String>,
        experiment_selection_stage_id: impl Into<String>,
        execution_stage_id: Option<String>,
        observation_stage_id: Option<String>,
        trace: &ScientificInvestigationTrace,
    ) -> Result<Self, InvestigationPlanningBridgeError> {
        let receipt = Self {
            receipt_version: VERSION.into(),
            investigation_id: investigation_id.into(),
            planning_trace_digest: planning_trace_digest.into(),
            model_council_artifact_digest: model_council_artifact_digest.into(),
            compiled_prediction_artifact_digest: compiled_prediction_artifact_digest.into(),
            prospective_commitment_digest: prospective_commitment_digest.into(),
            experiment_selection_artifact_digest: experiment_selection_artifact_digest.into(),
            prediction_stage_id: prediction_stage_id.into(),
            experiment_selection_stage_id: experiment_selection_stage_id.into(),
            execution_stage_id,
            observation_stage_id,
            receipt_digest: String::new(),
        };
        receipt.verify_against_trace(trace)?;
        let mut receipt = receipt;
        receipt.receipt_digest = receipt.compute_digest();
        Ok(receipt)
    }

    pub fn verify(&self) -> Result<(), InvestigationPlanningBridgeError> {
        if self.receipt_version != VERSION
            || self.investigation_id.trim().is_empty()
            || !valid_digest(&self.planning_trace_digest)
            || !valid_digest(&self.model_council_artifact_digest)
            || !valid_digest(&self.compiled_prediction_artifact_digest)
            || !valid_digest(&self.prospective_commitment_digest)
            || !valid_digest(&self.experiment_selection_artifact_digest)
            || self.prediction_stage_id.trim().is_empty()
            || self.experiment_selection_stage_id.trim().is_empty()
            || !valid_digest(&self.receipt_digest)
        {
            return Err(InvestigationPlanningBridgeError::InvalidReceipt);
        }
        if self.receipt_digest != self.compute_digest() {
            return Err(InvestigationPlanningBridgeError::ReceiptDigestMismatch);
        }
        Ok(())
    }

    pub fn verify_against_trace(
        &self,
        trace: &ScientificInvestigationTrace,
    ) -> Result<(), InvestigationPlanningBridgeError> {
        self.verify_without_receipt_digest()?;
        if self.investigation_id != trace.investigation_id {
            return Err(InvestigationPlanningBridgeError::InvestigationMismatch);
        }
        if self.planning_trace_digest != trace.trace_digest {
            return Err(InvestigationPlanningBridgeError::PlanningTraceMismatch);
        }

        let prediction = trace
            .stages
            .iter()
            .find(|s| s.stage_id == self.prediction_stage_id)
            .ok_or(InvestigationPlanningBridgeError::StageMissing)?;
        if prediction.kind != InvestigationStageKind::ProspectivePrediction {
            return Err(InvestigationPlanningBridgeError::StageKindMismatch);
        }
        if prediction.artifact.artifact_digest != self.prospective_commitment_digest {
            return Err(InvestigationPlanningBridgeError::CommitmentMismatch);
        }

        let selection = trace
            .stages
            .iter()
            .find(|s| s.stage_id == self.experiment_selection_stage_id)
            .ok_or(InvestigationPlanningBridgeError::StageMissing)?;
        if selection.kind != InvestigationStageKind::ExperimentSelection {
            return Err(InvestigationPlanningBridgeError::StageKindMismatch);
        }
        if selection.artifact.artifact_digest != self.experiment_selection_artifact_digest {
            return Err(InvestigationPlanningBridgeError::PlanningTraceMismatch);
        }

        if let Some(execution_id) = &self.execution_stage_id {
            let execution = trace
                .stages
                .iter()
                .find(|s| &s.stage_id == execution_id)
                .ok_or(InvestigationPlanningBridgeError::StageMissing)?;
            if execution.kind != InvestigationStageKind::ExternalExecution {
                return Err(InvestigationPlanningBridgeError::StageKindMismatch);
            }
            if trace
                .stages
                .iter()
                .position(|s| s.stage_id == *execution_id)
                .unwrap()
                <= trace
                    .stages
                    .iter()
                    .position(|s| s.stage_id == self.prediction_stage_id)
                    .unwrap()
            {
                return Err(InvestigationPlanningBridgeError::ExecutionBeforeCommitment);
            }
        }

        if let Some(observation_id) = &self.observation_stage_id {
            let observation = trace
                .stages
                .iter()
                .find(|s| &s.stage_id == observation_id)
                .ok_or(InvestigationPlanningBridgeError::StageMissing)?;
            if observation.kind != InvestigationStageKind::Observation {
                return Err(InvestigationPlanningBridgeError::StageKindMismatch);
            }
            let observation_pos = trace
                .stages
                .iter()
                .position(|s| s.stage_id == *observation_id)
                .unwrap();
            let execution_pos = self
                .execution_stage_id
                .as_ref()
                .and_then(|id| trace.stages.iter().position(|s| s.stage_id == *id))
                .ok_or(InvestigationPlanningBridgeError::ObservationBeforeExecution)?;
            if observation_pos <= execution_pos {
                return Err(InvestigationPlanningBridgeError::ObservationBeforeExecution);
            }
        }

        Ok(())
    }

    fn verify_without_receipt_digest(
        &self,
    ) -> Result<(), InvestigationPlanningBridgeError> {
        if self.receipt_version != VERSION
            || self.investigation_id.trim().is_empty()
            || !valid_digest(&self.planning_trace_digest)
            || !valid_digest(&self.model_council_artifact_digest)
            || !valid_digest(&self.compiled_prediction_artifact_digest)
            || !valid_digest(&self.prospective_commitment_digest)
            || !valid_digest(&self.experiment_selection_artifact_digest)
            || self.prediction_stage_id.trim().is_empty()
            || self.experiment_selection_stage_id.trim().is_empty()
        {
            return Err(InvestigationPlanningBridgeError::InvalidReceipt);
        }
        Ok(())
    }

    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(DOMAIN);
        for value in [
            &self.receipt_version,
            &self.investigation_id,
            &self.planning_trace_digest,
            &self.model_council_artifact_digest,
            &self.compiled_prediction_artifact_digest,
            &self.prospective_commitment_digest,
            &self.experiment_selection_artifact_digest,
            &self.prediction_stage_id,
            &self.experiment_selection_stage_id,
        ] {
            put(&mut h, value);
        }
        put(&mut h, self.execution_stage_id.as_deref().unwrap_or(""));
        put(&mut h, self.observation_stage_id.as_deref().unwrap_or(""));
        format!("sha256:{:x}", h.finalize())
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
    use crate::scientific_investigation_trace::{
        InvestigationArtifactRef, InvestigationOutcome, InvestigationStage,
    };

    fn digest(ch: char) -> String {
        format!("sha256:{}", ch.to_string().repeat(64))
    }

    fn stage(id: &str, kind: InvestigationStageKind, artifact: &str, parents: &[&str]) -> InvestigationStage {
        InvestigationStage {
            stage_id: id.into(),
            kind,
            artifact: InvestigationArtifactRef {
                artifact_id: format!("artifact:{id}"),
                artifact_digest: artifact.into(),
            },
            outcome: InvestigationOutcome::Pending,
            parent_stage_ids: parents.iter().map(|p| (*p).into()).collect(),
        }
    }

    fn trace() -> ScientificInvestigationTrace {
        ScientificInvestigationTrace::new(
            "investigation:003ab",
            None,
            vec![
                stage("council", InvestigationStageKind::ModelCouncilSelection, &digest('b'), &[]),
                stage("prediction", InvestigationStageKind::ProspectivePrediction, &digest('c'), &["council"]),
                stage("selection", InvestigationStageKind::ExperimentSelection, &digest('d'), &["prediction"]),
                stage("execution", InvestigationStageKind::ExternalExecution, &digest('e'), &["selection"]),
                stage("observation", InvestigationStageKind::Observation, &digest('f'), &["execution"]),
            ],
        )
        .unwrap()
    }

    fn receipt(trace: &ScientificInvestigationTrace) -> InvestigationPlanningBridgeReceipt {
        InvestigationPlanningBridgeReceipt::new(
            "investigation:003ab",
            trace.trace_digest.clone(),
            digest('b'),
            digest('a'),
            digest('c'),
            digest('d'),
            "prediction",
            "selection",
            Some("execution".into()),
            Some("observation".into()),
            trace,
        )
        .unwrap()
    }

    #[test]
    fn exact_commitment_is_bound_to_prediction_stage() {
        let trace = trace();
        assert!(receipt(&trace).verify_against_trace(&trace).is_ok());
    }

    #[test]
    fn commitment_substitution_is_rejected() {
        let trace = trace();
        let mut r = receipt(&trace);
        r.prospective_commitment_digest = digest('9');
        assert_eq!(
            r.verify_against_trace(&trace),
            Err(InvestigationPlanningBridgeError::CommitmentMismatch)
        );
    }

    #[test]
    fn selection_substitution_is_rejected() {
        let trace = trace();
        let mut r = receipt(&trace);
        r.experiment_selection_artifact_digest = digest('9');
        assert_eq!(
            r.verify_against_trace(&trace),
            Err(InvestigationPlanningBridgeError::PlanningTraceMismatch)
        );
    }

    #[test]
    fn trace_substitution_is_rejected() {
        let trace = trace();
        let mut r = receipt(&trace);
        let other = ScientificInvestigationTrace::new(
            "investigation:other",
            None,
            vec![stage("hypothesis", InvestigationStageKind::Hypothesis, &digest('a'), &[])],
        )
        .unwrap();
        r.planning_trace_digest = other.trace_digest.clone();
        assert_eq!(
            r.verify_against_trace(&trace),
            Err(InvestigationPlanningBridgeError::PlanningTraceMismatch)
        );
    }

    #[test]
    fn observation_cannot_precede_execution() {
        let trace = trace();
        let mut r = receipt(&trace);
        r.execution_stage_id = Some("prediction".into());
        assert_eq!(
            r.verify_against_trace(&trace),
            Err(InvestigationPlanningBridgeError::StageKindMismatch)
        );
    }

    #[test]
    fn receipt_digest_changes_when_compiled_prediction_changes() {
        let trace = trace();
        let r = receipt(&trace);
        let mut altered = r.clone();
        altered.compiled_prediction_artifact_digest = digest('9');
        assert_ne!(r.receipt_digest, altered.compute_digest());
    }

    #[test]
    fn serde_roundtrip() {
        let trace = trace();
        let r = receipt(&trace);
        let encoded = serde_json::to_vec(&r).unwrap();
        assert_eq!(serde_json::from_slice::<InvestigationPlanningBridgeReceipt>(&encoded).unwrap(), r);
    }
}
