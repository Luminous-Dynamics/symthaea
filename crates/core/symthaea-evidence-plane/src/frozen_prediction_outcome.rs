//! Exact comparison of external observations against a frozen prospective prediction.
//!
//! This module deliberately does not regenerate a prediction. It binds the
//! observation to the exact commitment digest created before execution and
//! records only an analysis disposition.

use crate::scientific_investigation_trace::{InvestigationStageKind, ScientificInvestigationTrace};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const VERSION: &str = "1.0.0";
const DOMAIN: &[u8] = b"symthaea:frozen-prediction-comparison:v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionOutcomeDisposition {
    Supports,
    Contradicts,
    Null,
    Inconclusive,
    Partial,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenPredictionOutcomeReceipt {
    pub receipt_version: String,
    pub investigation_id: String,
    pub trace_digest: String,
    pub prediction_stage_id: String,
    pub prospective_commitment_digest: String,
    pub observation_stage_id: String,
    pub observation_artifact_digest: String,
    pub comparator_artifact_digest: String,
    pub disposition: PredictionOutcomeDisposition,
    pub receipt_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrozenPredictionComparisonError {
    InvalidReceipt,
    InvalidDigest,
    InvestigationMismatch,
    TraceMismatch,
    PredictionStageMissing,
    PredictionStageKindMismatch,
    ObservationStageMissing,
    ObservationStageKindMismatch,
    CommitmentMismatch,
    ObservationMismatch,
    ComparatorMismatch,
    ObservationBeforeExecution,
    ObservationBeforePrediction,
    ReceiptDigestMismatch,
}

impl FrozenPredictionOutcomeReceipt {
    pub fn new(
        trace: &ScientificInvestigationTrace,
        prediction_stage_id: impl Into<String>,
        observation_stage_id: impl Into<String>,
        prospective_commitment_digest: impl Into<String>,
        comparator_artifact_digest: impl Into<String>,
        disposition: PredictionOutcomeDisposition,
    ) -> Result<Self, FrozenPredictionComparisonError> {
        let prediction_stage_id = prediction_stage_id.into();
        let observation_stage_id = observation_stage_id.into();
        let receipt = Self {
            receipt_version: VERSION.into(),
            investigation_id: trace.investigation_id.clone(),
            trace_digest: trace.trace_digest.clone(),
            prediction_stage_id,
            prospective_commitment_digest: prospective_commitment_digest.into(),
            observation_stage_id,
            observation_artifact_digest: String::new(),
            comparator_artifact_digest: comparator_artifact_digest.into(),
            disposition,
            receipt_digest: String::new(),
        };
        let observation = trace.stages.iter()
            .find(|s| s.stage_id == receipt.observation_stage_id)
            .ok_or(FrozenPredictionComparisonError::ObservationStageMissing)?;
        let mut receipt = receipt;
        receipt.observation_artifact_digest = observation.artifact.artifact_digest.clone();
        receipt.verify_against_trace(trace)?;
        receipt.receipt_digest = receipt.compute_digest();
        Ok(receipt)
    }

    pub fn verify(&self) -> Result<(), FrozenPredictionComparisonError> {
        if self.receipt_version != VERSION
            || self.investigation_id.trim().is_empty()
            || !valid_digest(&self.trace_digest)
            || !valid_digest(&self.prospective_commitment_digest)
            || !valid_digest(&self.observation_artifact_digest)
            || !valid_digest(&self.comparator_artifact_digest)
            || !valid_digest(&self.receipt_digest)
            || self.prediction_stage_id.trim().is_empty()
            || self.observation_stage_id.trim().is_empty()
        { return Err(FrozenPredictionComparisonError::InvalidReceipt); }
        if self.receipt_digest != self.compute_digest() {
            return Err(FrozenPredictionComparisonError::ReceiptDigestMismatch);
        }
        Ok(())
    }

    pub fn verify_against_trace(&self, trace: &ScientificInvestigationTrace)
        -> Result<(), FrozenPredictionComparisonError> {
        self.verify_without_digest()?;
        if self.investigation_id != trace.investigation_id {
            return Err(FrozenPredictionComparisonError::InvestigationMismatch);
        }
        if self.trace_digest != trace.trace_digest {
            return Err(FrozenPredictionComparisonError::TraceMismatch);
        }
        let p = trace.stages.iter().find(|s| s.stage_id == self.prediction_stage_id)
            .ok_or(FrozenPredictionComparisonError::PredictionStageMissing)?;
        if p.kind != InvestigationStageKind::ProspectivePrediction {
            return Err(FrozenPredictionComparisonError::PredictionStageKindMismatch);
        }
        if p.artifact.artifact_digest != self.prospective_commitment_digest {
            return Err(FrozenPredictionComparisonError::CommitmentMismatch);
        }
        let o = trace.stages.iter().find(|s| s.stage_id == self.observation_stage_id)
            .ok_or(FrozenPredictionComparisonError::ObservationStageMissing)?;
        if o.kind != InvestigationStageKind::Observation {
            return Err(FrozenPredictionComparisonError::ObservationStageKindMismatch);
        }
        if o.artifact.artifact_digest != self.observation_artifact_digest {
            return Err(FrozenPredictionComparisonError::ObservationMismatch);
        }
        let pp = trace.stages.iter().position(|s| s.stage_id == self.prediction_stage_id).unwrap();
        let op = trace.stages.iter().position(|s| s.stage_id == self.observation_stage_id).unwrap();
        if pp >= op { return Err(FrozenPredictionComparisonError::ObservationBeforePrediction); }
        let ep = trace.stages.iter().position(|s| s.kind == InvestigationStageKind::ExternalExecution)
            .ok_or(FrozenPredictionComparisonError::ObservationBeforeExecution)?;
        if ep >= op { return Err(FrozenPredictionComparisonError::ObservationBeforeExecution); }
        Ok(())
    }

    fn verify_without_digest(&self) -> Result<(), FrozenPredictionComparisonError> {
        if self.receipt_version != VERSION || self.investigation_id.trim().is_empty()
            || !valid_digest(&self.trace_digest)
            || !valid_digest(&self.prospective_commitment_digest)
            || !valid_digest(&self.observation_artifact_digest)
            || !valid_digest(&self.comparator_artifact_digest)
            || self.prediction_stage_id.trim().is_empty()
            || self.observation_stage_id.trim().is_empty()
        { return Err(FrozenPredictionComparisonError::InvalidReceipt); }
        Ok(())
    }

    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(DOMAIN);
        for value in [
            &self.receipt_version, &self.investigation_id, &self.trace_digest,
            &self.prediction_stage_id, &self.prospective_commitment_digest,
            &self.observation_stage_id, &self.observation_artifact_digest,
            &self.comparator_artifact_digest,
        ] { put(&mut h, value); }
        put(&mut h, &format!("{:?}", self.disposition));
        format!("sha256:{:x}", h.finalize())
    }
}
fn valid_digest(v: &str) -> bool {
    v.len() == 71 && v.starts_with("sha256:") && v.as_bytes()[7..].iter().all(|b| b.is_ascii_hexdigit())
}
fn put(h: &mut Sha256, v: &str) { h.update((v.len() as u64).to_be_bytes()); h.update(v.as_bytes()); }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scientific_investigation_trace::{InvestigationArtifactRef, InvestigationOutcome, InvestigationStage};
    fn d(c: char)->String { format!("sha256:{}", c.to_string().repeat(64)) }
    fn s(id:&str,k:InvestigationStageKind,a:char,p:&[&str])->InvestigationStage {
        InvestigationStage { stage_id:id.into(), kind:k, artifact:InvestigationArtifactRef{artifact_id:id.into(),artifact_digest:d(a)}, outcome:InvestigationOutcome::Pending, parent_stage_ids:p.iter().map(|x|x.to_string()).collect() }
    }
    fn trace()->ScientificInvestigationTrace {
        ScientificInvestigationTrace::new("investigation:003ac",None,vec![
            s("prediction",InvestigationStageKind::ProspectivePrediction,'a',&[]),
            s("selection",InvestigationStageKind::ExperimentSelection,'b',&["prediction"]),
            s("execution",InvestigationStageKind::ExternalExecution,'c',&["selection"]),
            s("observation",InvestigationStageKind::Observation,'d',&["execution"]),
        ]).unwrap()
    }
    fn receipt(t:&ScientificInvestigationTrace)->FrozenPredictionOutcomeReceipt {
        FrozenPredictionOutcomeReceipt::new(t,"prediction","observation",d('a'),d('e'),PredictionOutcomeDisposition::Contradicts).unwrap()
    }
    #[test] fn exact_frozen_commitment_is_required(){ let t=trace(); assert!(receipt(&t).verify_against_trace(&t).is_ok()); }
    #[test] fn commitment_substitution_rejected(){ let t=trace(); let mut r=receipt(&t); r.prospective_commitment_digest=d('9'); assert_eq!(r.verify_against_trace(&t),Err(FrozenPredictionComparisonError::CommitmentMismatch)); }
    #[test] fn observation_substitution_rejected(){ let t=trace(); let mut r=receipt(&t); r.observation_artifact_digest=d('9'); assert_eq!(r.verify_against_trace(&t),Err(FrozenPredictionComparisonError::ObservationMismatch)); }
    #[test] fn trace_substitution_rejected(){ let t=trace(); let mut r=receipt(&t); r.trace_digest=d('9'); assert_eq!(r.verify_against_trace(&t),Err(FrozenPredictionComparisonError::TraceMismatch)); }
    #[test] fn comparator_substitution_changes_receipt_identity(){ let t=trace(); let r=receipt(&t); let mut x=r.clone(); x.comparator_artifact_digest=d('9'); assert_ne!(r.receipt_digest,x.compute_digest()); }
    #[test] fn negative_and_null_are_first_class(){ let t=trace(); for dpos in [PredictionOutcomeDisposition::Contradicts,PredictionOutcomeDisposition::Null,PredictionOutcomeDisposition::Inconclusive,PredictionOutcomeDisposition::Partial] { let r=FrozenPredictionOutcomeReceipt::new(&t,"prediction","observation",d('a'),d('e'),dpos).unwrap(); assert!(r.verify_against_trace(&t).is_ok()); } }
    #[test] fn serde_roundtrip(){ let t=trace(); let r=receipt(&t); let x=serde_json::to_vec(&r).unwrap(); assert_eq!(serde_json::from_slice::<FrozenPredictionOutcomeReceipt>(&x).unwrap(),r); }
}
