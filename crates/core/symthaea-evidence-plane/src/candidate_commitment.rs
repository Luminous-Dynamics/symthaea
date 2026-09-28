// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Binds a materialized test candidate to a prospective prediction only.
use serde::{Deserialize, Serialize};\nuse sha2::Digest;
use crate::prospective::{CommitmentError,ProspectivePredictionCommitment,ProspectiveProvenance};
use crate::temporal_test_candidate::TemporalTestCandidateSpec;

#[derive(Debug,Clone,PartialEq,Serialize,Deserialize)]
pub struct CandidatePredictionBinding {
    pub candidate_id:String,
    pub source_candidate_id:String,
    pub test_specification_id:String,
    pub measurement_specification_id:String,
    pub prediction_payload:Vec<u8>,
}

#[test]
fn envelope_exposes_reproducible_binding_digest() {
    let c = candidate("c1");
    let envelope = commit_candidate_envelope(&c,"challenge-v1","criteria-v1","mapping-v1","actor-v1","2026-09-28T09:00:00Z",provenance(),b"forecast").unwrap();
    let binding = CandidatePredictionBinding { candidate_id:c.candidate_id.clone(), source_candidate_id:c.source_candidate_id.clone(), test_specification_id:c.test_specification_id.clone(), measurement_specification_id:c.measurement_specification_id.clone(), prediction_payload:b"forecast".to_vec() };
    assert_eq!(envelope.binding_digest, binding.digest().unwrap());
    assert!(envelope.verify_binding(&binding).unwrap());
}
#[test]
fn envelope_detects_tampered_binding() {
    let c = candidate("c1");
    let envelope = commit_candidate_envelope(&c,"challenge-v1","criteria-v1","mapping-v1","actor-v1","2026-09-28T09:00:00Z",provenance(),b"forecast").unwrap();
    let mut binding = CandidatePredictionBinding { candidate_id:c.candidate_id, source_candidate_id:c.source_candidate_id, test_specification_id:c.test_specification_id, measurement_specification_id:c.measurement_specification_id, prediction_payload:b"tampered".to_vec() };
    assert!(!envelope.verify_binding(&binding).unwrap());
    binding.candidate_id = "c2".into();
    assert!(!envelope.verify_binding(&binding).unwrap());
}
#[derive(Debug,Clone,PartialEq,Eq)]
pub enum BindingError{EmptyPredictionPayload,MissingBindingField(&'static str),Serialization,Commitment(CommitmentError)}
impl From<CommitmentError> for BindingError{fn from(e:CommitmentError)->Self{Self::Commitment(e)}}
/// Commit caller-authored prediction bytes with exact candidate and test IDs.
/// This emits no experimental observation or criterion evidence.
pub fn commit_candidate(candidate:&TemporalTestCandidateSpec,challenge_id:&str,criteria_generation:&str,mapping_generation:&str,actor_id:&str,created_at:&str,provenance:ProspectiveProvenance,prediction_payload:&[u8])->Result<ProspectivePredictionCommitment,BindingError>{
    if prediction_payload.is_empty(){return Err(BindingError::EmptyPredictionPayload);}
    let binding=CandidatePredictionBinding{candidate_id:candidate.candidate_id.clone(),source_candidate_id:candidate.source_candidate_id.clone(),test_specification_id:candidate.test_specification_id.clone(),measurement_specification_id:candidate.measurement_specification_id.clone(),prediction_payload:prediction_payload.to_vec()};
    let bytes=binding.canonical_bytes()?;
    Ok(ProspectivePredictionCommitment::commit(challenge_id,criteria_generation,mapping_generation,actor_id,created_at,provenance,&bytes)?)
}

#[cfg(test)]
mod tests {
 use super::*;
 fn candidate(id:&str)->TemporalTestCandidateSpec { TemporalTestCandidateSpec{candidate_id:id.into(),left_model_id:"m1".into(),right_model_id:"m2".into(),left_lineage:"l1".into(),right_lineage:"l2".into(),outcome_id:"o1".into(),horizon_seconds:3.0,advances_unmet_predicates:["replication".to_string()].into_iter().collect(),test_specification_id:"test-v1".into(),measurement_specification_id:"measure-v1".into(),estimated_cost:1.0,pragmatic_risk:0.2,source_candidate_id:"source-v1".into()} }
 fn provenance()->ProspectiveProvenance { ProspectiveProvenance::new("sha256:input","sha256:artifact","lineage:m1+m2","2026-09-28T08:00:00Z","2026-09-28T09:00:00Z").unwrap() }
 fn commit(c:&TemporalTestCandidateSpec,payload:&[u8])->Result<ProspectivePredictionCommitment,BindingError>{commit_candidate(c,"challenge-v1","criteria-v1","mapping-v1","actor-v1","2026-09-28T09:00:00Z",provenance(),payload)}
 #[test] fn binds_exact_candidate_and_prediction(){let a=commit(&candidate("c1"),b"forecast").unwrap();let b=commit(&candidate("c2"),b"forecast").unwrap();assert_ne!(a.payload_digest(),b.payload_digest());assert_eq!(a,commit(&candidate("c1"),b"forecast").unwrap());}
 #[test] fn rejects_empty_prediction(){assert_eq!(commit(&candidate("c1"),b""),Err(BindingError::EmptyPredictionPayload));}
 #[test] fn supersession_is_a_new_commitment_with_parent(){let a=commit(&candidate("c1"),b"forecast-v1").unwrap();let b=a.supersede("actor-v2","2026-09-28T10:00:00Z",b"forecast-v2",provenance()).unwrap();assert_ne!(a.event_id(),b.event_id());assert_eq!(b.parent_event_ids(),&[a.event_id().to_string()]);}
 #[test] fn cutoff_order_is_enforced(){let bad=ProspectiveProvenance::new("input","artifact","lineage","2026-09-28T10:00:00Z","2026-09-28T09:00:00Z");assert!(matches!(bad,Err(CommitmentError::ExposureBeforeKnowledgeCutoff)));}
}

#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize)]
pub struct CandidatePredictionCommitment {
    pub commitment: ProspectivePredictionCommitment,
    pub candidate_id: String,
    pub source_candidate_id: String,
    pub test_specification_id: String,
    pub measurement_specification_id: String,
    pub binding_digest: String,
}
impl CandidatePredictionBinding {
    pub fn validate(&self) -> Result<(), BindingError> {
        for (name, value) in [("candidate_id",&self.candidate_id),("source_candidate_id",&self.source_candidate_id),("test_specification_id",&self.test_specification_id),("measurement_specification_id",&self.measurement_specification_id)] {
            if value.trim().is_empty() { return Err(BindingError::MissingBindingField(name)); }
        }
        if self.prediction_payload.is_empty() { return Err(BindingError::EmptyPredictionPayload); }
        Ok(())
    }
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, BindingError> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|_| BindingError::Serialization)
    }
    pub fn digest(&self) -> Result<String, BindingError> {
        let bytes=self.canonical_bytes()?;
        let mut h=sha2::Sha256::new();
        h.update(b"symthaea:candidate-prediction-binding:v1\0");
        h.update((bytes.len() as u64).to_be_bytes());
        h.update(bytes);
        Ok(format!("sha256:{:x}",h.finalize()))
    }
}
impl CandidatePredictionCommitment {
    pub fn verify_binding(&self, binding:&CandidatePredictionBinding) -> Result<bool, BindingError> {
        Ok(self.binding_digest == binding.digest()? && self.candidate_id == binding.candidate_id && self.source_candidate_id == binding.source_candidate_id && self.test_specification_id == binding.test_specification_id && self.measurement_specification_id == binding.measurement_specification_id)
    }
}
pub fn commit_candidate_envelope(candidate:&TemporalTestCandidateSpec,challenge_id:&str,criteria_generation:&str,mapping_generation:&str,actor_id:&str,created_at:&str,provenance:ProspectiveProvenance,prediction_payload:&[u8])->Result<CandidatePredictionCommitment,BindingError>{
    let binding=CandidatePredictionBinding{candidate_id:candidate.candidate_id.clone(),source_candidate_id:candidate.source_candidate_id.clone(),test_specification_id:candidate.test_specification_id.clone(),measurement_specification_id:candidate.measurement_specification_id.clone(),prediction_payload:prediction_payload.to_vec()};
    let bytes=binding.canonical_bytes()?;
    let digest=binding.digest()?;
    let commitment=ProspectivePredictionCommitment::commit(challenge_id,criteria_generation,mapping_generation,actor_id,created_at,provenance,&bytes)?;
    Ok(CandidatePredictionCommitment{commitment,candidate_id:binding.candidate_id,source_candidate_id:binding.source_candidate_id,test_specification_id:binding.test_specification_id,measurement_specification_id:binding.measurement_specification_id,binding_digest:digest})
}
