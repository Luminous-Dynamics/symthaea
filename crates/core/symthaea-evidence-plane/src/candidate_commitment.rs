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
