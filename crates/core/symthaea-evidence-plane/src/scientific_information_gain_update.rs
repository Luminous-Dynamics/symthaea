//! Post-observation information-gain update for a scientific investigation.
//!
//! This receipt closes the loop after a frozen prediction is compared with an
//! external observation. It binds an immutable comparison to a *new* analysis
//! and planning state without treating information gain as evidence or allowing
//! the historical prediction/observation/comparison artifacts to be rewritten.

use crate::frozen_prediction_outcome::PredictionOutcomeDisposition;
use crate::scientific_investigation_trace::{
    InvestigationStageKind, ScientificInvestigationTrace,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const VERSION: &str = "1.0.0";
const DOMAIN: &[u8] = b"symthaea:scientific-information-gain-update:v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificInformationGainUpdateReceipt {
    pub receipt_version: String,
    pub investigation_id: String,
    pub trace_digest: String,
    pub frozen_comparison_receipt_digest: String,
    pub prediction_stage_id: String,
    pub prospective_commitment_digest: String,
    pub execution_stage_id: String,
    pub observation_stage_id: String,
    pub observation_artifact_digest: String,
    pub disposition: PredictionOutcomeDisposition,
    pub pre_update_state_digest: String,
    pub information_gain_artifact_digest: String,
    pub post_update_state_digest: String,
    pub next_planning_input_digest: String,
    pub update_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScientificInformationGainUpdateError {
    InvalidReceipt,
    InvalidDigest,
    InvestigationMismatch,
    TraceMismatch,
    PredictionStageMissing,
    PredictionStageKindMismatch,
    CommitmentMismatch,
    ExecutionStageMissing,
    ExecutionStageKindMismatch,
    ObservationStageMissing,
    ObservationStageKindMismatch,
    ObservationMismatch,
    ExecutionNotObservationParent,
    ExecutionBeforePrediction,
    ObservationBeforeExecution,
    UpdateDigestMismatch,
}

impl ScientificInformationGainUpdateReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        trace: &ScientificInvestigationTrace,
        frozen_comparison_receipt_digest: impl Into<String>,
        prediction_stage_id: impl Into<String>,
        prospective_commitment_digest: impl Into<String>,
        execution_stage_id: impl Into<String>,
        observation_stage_id: impl Into<String>,
        pre_update_state_digest: impl Into<String>,
        information_gain_artifact_digest: impl Into<String>,
        post_update_state_digest: impl Into<String>,
        next_planning_input_digest: impl Into<String>,
        disposition: PredictionOutcomeDisposition,
    ) -> Result<Self, ScientificInformationGainUpdateError> {
        let observation_stage_id = observation_stage_id.into();
        let observation = trace
            .stages
            .iter()
            .find(|s| s.stage_id == observation_stage_id)
            .ok_or(ScientificInformationGainUpdateError::ObservationStageMissing)?;

        let receipt = Self {
            receipt_version: VERSION.into(),
            investigation_id: trace.investigation_id.clone(),
            trace_digest: trace.trace_digest.clone(),
            frozen_comparison_receipt_digest: frozen_comparison_receipt_digest.into(),
            prediction_stage_id: prediction_stage_id.into(),
            prospective_commitment_digest: prospective_commitment_digest.into(),
            execution_stage_id: execution_stage_id.into(),
            observation_stage_id,
            observation_artifact_digest: observation.artifact.artifact_digest.clone(),
            disposition,
            pre_update_state_digest: pre_update_state_digest.into(),
            information_gain_artifact_digest: information_gain_artifact_digest.into(),
            post_update_state_digest: post_update_state_digest.into(),
            next_planning_input_digest: next_planning_input_digest.into(),
            update_digest: String::new(),
        };

        receipt.verify_against_trace(trace)?;
        let mut receipt = receipt;
        receipt.update_digest = receipt.compute_digest();
        Ok(receipt)
    }

    pub fn verify(&self) -> Result<(), ScientificInformationGainUpdateError> {
        self.verify_without_digest()?;
        if self.update_digest != self.compute_digest() {
            return Err(ScientificInformationGainUpdateError::UpdateDigestMismatch);
        }
        Ok(())
    }

    pub fn verify_against_trace(
        &self,
        trace: &ScientificInvestigationTrace,
    ) -> Result<(), ScientificInformationGainUpdateError> {
        self.verify_without_digest()?;

        if self.investigation_id != trace.investigation_id {
            return Err(ScientificInformationGainUpdateError::InvestigationMismatch);
        }
        if self.trace_digest != trace.trace_digest {
            return Err(ScientificInformationGainUpdateError::TraceMismatch);
        }

        let prediction = trace
            .stages
            .iter()
            .find(|s| s.stage_id == self.prediction_stage_id)
            .ok_or(ScientificInformationGainUpdateError::PredictionStageMissing)?;
        if prediction.kind != InvestigationStageKind::ProspectivePrediction {
            return Err(ScientificInformationGainUpdateError::PredictionStageKindMismatch);
        }
        if prediction.artifact.artifact_digest != self.prospective_commitment_digest {
            return Err(ScientificInformationGainUpdateError::CommitmentMismatch);
        }

        let execution = trace
            .stages
            .iter()
            .find(|s| s.stage_id == self.execution_stage_id)
            .ok_or(ScientificInformationGainUpdateError::ExecutionStageMissing)?;
        if execution.kind != InvestigationStageKind::ExternalExecution {
            return Err(ScientificInformationGainUpdateError::ExecutionStageKindMismatch);
        }

        let observation = trace
            .stages
            .iter()
            .find(|s| s.stage_id == self.observation_stage_id)
            .ok_or(ScientificInformationGainUpdateError::ObservationStageMissing)?;
        if observation.kind != InvestigationStageKind::Observation {
            return Err(ScientificInformationGainUpdateError::ObservationStageKindMismatch);
        }
        if observation.artifact.artifact_digest != self.observation_artifact_digest {
            return Err(ScientificInformationGainUpdateError::ObservationMismatch);
        }
        if !observation
            .parent_stage_ids
            .iter()
            .any(|id| id == &self.execution_stage_id)
        {
            return Err(ScientificInformationGainUpdateError::ExecutionNotObservationParent);
        }

        let prediction_pos = trace
            .stages
            .iter()
            .position(|s| s.stage_id == self.prediction_stage_id)
            .expect("prediction stage checked above");
        let execution_pos = trace
            .stages
            .iter()
            .position(|s| s.stage_id == self.execution_stage_id)
            .expect("execution stage checked above");
        let observation_pos = trace
            .stages
            .iter()
            .position(|s| s.stage_id == self.observation_stage_id)
            .expect("observation stage checked above");

        if execution_pos <= prediction_pos {
            return Err(ScientificInformationGainUpdateError::ExecutionBeforePrediction);
        }
        if observation_pos <= execution_pos {
            return Err(ScientificInformationGainUpdateError::ObservationBeforeExecution);
        }

        Ok(())
    }

    fn verify_without_digest(&self) -> Result<(), ScientificInformationGainUpdateError> {
        if self.receipt_version != VERSION
            || self.investigation_id.trim().is_empty()
            || !valid_digest(&self.trace_digest)
            || !valid_digest(&self.frozen_comparison_receipt_digest)
            || self.prediction_stage_id.trim().is_empty()
            || !valid_digest(&self.prospective_commitment_digest)
            || self.execution_stage_id.trim().is_empty()
            || self.observation_stage_id.trim().is_empty()
            || !valid_digest(&self.observation_artifact_digest)
            || !valid_digest(&self.pre_update_state_digest)
            || !valid_digest(&self.information_gain_artifact_digest)
            || !valid_digest(&self.post_update_state_digest)
            || !valid_digest(&self.next_planning_input_digest)
            || !valid_digest(&self.update_digest)
        {
            return Err(ScientificInformationGainUpdateError::InvalidReceipt);
        }
        Ok(())
    }

    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(DOMAIN);
        for value in [
            &self.receipt_version,
            &self.investigation_id,
            &self.trace_digest,
            &self.frozen_comparison_receipt_digest,
            &self.prediction_stage_id,
            &self.prospective_commitment_digest,
            &self.execution_stage_id,
            &self.observation_stage_id,
            &self.observation_artifact_digest,
            &self.pre_update_state_digest,
            &self.information_gain_artifact_digest,
            &self.post_update_state_digest,
            &self.next_planning_input_digest,
        ] {
            put(&mut h, value);
        }
        put(&mut h, &format!("{:?}", self.disposition));
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

    fn stage(
        id: &str,
        kind: InvestigationStageKind,
        artifact: char,
        parents: &[&str],
    ) -> InvestigationStage {
        InvestigationStage {
            stage_id: id.into(),
            kind,
            artifact: InvestigationArtifactRef {
                artifact_id: format!("artifact:{id}"),
                artifact_digest: digest(artifact),
            },
            outcome: InvestigationOutcome::Pending,
            parent_stage_ids: parents.iter().map(|p| (*p).into()).collect(),
        }
    }

    fn trace() -> ScientificInvestigationTrace {
        ScientificInvestigationTrace::new(
            "investigation:003ad",
            None,
            vec![
                stage("prediction", InvestigationStageKind::ProspectivePrediction, 'a', &[]),
                stage("selection", InvestigationStageKind::ExperimentSelection, 'b', &["prediction"]),
                stage("execution", InvestigationStageKind::ExternalExecution, 'c', &["selection"]),
                stage("observation", InvestigationStageKind::Observation, 'd', &["execution"]),
            ],
        )
        .unwrap()
    }

    fn receipt(t: &ScientificInvestigationTrace) -> ScientificInformationGainUpdateReceipt {
        ScientificInformationGainUpdateReceipt::new(
            t,
            digest('e'),
            "prediction",
            digest('a'),
            "execution",
            "observation",
            digest('f'),
            digest('g'),
            digest('h'),
            digest('i'),
            PredictionOutcomeDisposition::Contradicts,
        )
        .unwrap()
    }

    #[test]
    fn exact_execution_parent_is_required() {
        let t = trace();
        assert!(receipt(&t).verify_against_trace(&t).is_ok());
    }

    #[test]
    fn unrelated_execution_is_rejected() {
        let mut t = trace();
        t.stages.insert(
            3,
            stage(
                "other-execution",
                InvestigationStageKind::ExternalExecution,
                'x',
                &["selection"],
            ),
        );
        t.stages[4].parent_stage_ids = vec!["other-execution".into()];
        assert_eq!(
            receipt(&t).verify_against_trace(&t),
            Err(ScientificInformationGainUpdateError::ExecutionNotObservationParent)
        );
    }

    #[test]
    fn comparison_substitution_changes_identity() {
        let t = trace();
        let r = receipt(&t);
        let mut x = r.clone();
        x.frozen_comparison_receipt_digest = digest('z');
        assert_ne!(r.update_digest, x.compute_digest());
    }

    #[test]
    fn pre_state_substitution_changes_identity() {
        let t = trace();
        let r = receipt(&t);
        let mut x = r.clone();
        x.pre_update_state_digest = digest('z');
        assert_ne!(r.update_digest, x.compute_digest());
    }

    #[test]
    fn information_gain_substitution_changes_identity() {
        let t = trace();
        let r = receipt(&t);
        let mut x = r.clone();
        x.information_gain_artifact_digest = digest('z');
        assert_ne!(r.update_digest, x.compute_digest());
    }

    #[test]
    fn post_state_substitution_changes_identity() {
        let t = trace();
        let r = receipt(&t);
        let mut x = r.clone();
        x.post_update_state_digest = digest('z');
        assert_ne!(r.update_digest, x.compute_digest());
    }

    #[test]
    fn next_planning_input_substitution_changes_identity() {
        let t = trace();
        let r = receipt(&t);
        let mut x = r.clone();
        x.next_planning_input_digest = digest('z');
        assert_ne!(r.update_digest, x.compute_digest());
    }

    #[test]
    fn historical_observation_substitution_is_rejected() {
        let t = trace();
        let mut r = receipt(&t);
        r.observation_artifact_digest = digest('z');
        assert_eq!(
            r.verify_against_trace(&t),
            Err(ScientificInformationGainUpdateError::ObservationMismatch)
        );
    }

    #[test]
    fn all_comparison_dispositions_remain_valid_inputs() {
        let t = trace();
        for disposition in [
            PredictionOutcomeDisposition::Supports,
            PredictionOutcomeDisposition::Contradicts,
            PredictionOutcomeDisposition::Null,
            PredictionOutcomeDisposition::Inconclusive,
            PredictionOutcomeDisposition::Partial,
        ] {
            let r = ScientificInformationGainUpdateReceipt::new(
                &t,
                digest('e'),
                "prediction",
                digest('a'),
                "execution",
                "observation",
                digest('f'),
                digest('g'),
                digest('h'),
                digest('i'),
                disposition,
            )
            .unwrap();
            assert!(r.verify_against_trace(&t).is_ok());
        }
    }

    #[test]
    fn serde_roundtrip_and_digest_verification() {
        let t = trace();
        let r = receipt(&t);
        assert!(r.verify().is_ok());
        let encoded = serde_json::to_vec(&r).unwrap();
        let decoded: ScientificInformationGainUpdateReceipt =
            serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, r);
        assert!(decoded.verify_against_trace(&t).is_ok());
    }
}
