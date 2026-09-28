// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Binds a materialized test candidate to a prospective prediction only.
use serde::{Deserialize, Serialize};
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
#[derive(Debug,Clone,PartialEq,Eq)]
pub enum BindingError{EmptyPredictionPayload,Commitment(CommitmentError)}
impl From<CommitmentError> for BindingError{fn from(e:CommitmentError)->Self{Self::Commitment(e)}}
/// Commit caller-authored prediction bytes with exact candidate and test IDs.
/// This emits no experimental observation or criterion evidence.
pub fn commit_candidate(candidate:&TemporalTestCandidateSpec,challenge_id:&str,criteria_generation:&str,mapping_generation:&str,actor_id:&str,created_at:&str,provenance:ProspectiveProvenance,prediction_payload:&[u8])->Result<ProspectivePredictionCommitment,BindingError>{
    if prediction_payload.is_empty(){return Err(BindingError::EmptyPredictionPayload);}
    let binding=CandidatePredictionBinding{candidate_id:candidate.candidate_id.clone(),source_candidate_id:candidate.source_candidate_id.clone(),test_specification_id:candidate.test_specification_id.clone(),measurement_specification_id:candidate.measurement_specification_id.clone(),prediction_payload:prediction_payload.to_vec()};
    let bytes=serde_json::to_vec(&binding).expect("serializing a concrete candidate binding cannot fail");
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
