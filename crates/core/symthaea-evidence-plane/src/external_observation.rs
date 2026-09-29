// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ingress for externally asserted observations. This records provenance only;
//! it does not execute experiments, assess truth, or promote evidence.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use crate::candidate_commitment::{CandidatePredictionBinding, CandidatePredictionCommitment, VerificationFailure};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all="snake_case")]
pub enum ObservationDisposition { Reported, Failed, NullResult, Contradictory, Inconclusive }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalObservationInput {
 pub observation_id:String, pub execution_id:String, pub observer_id:String, pub institution_id:String,
 pub observed_at:String, pub disposition:ObservationDisposition,
 /// Opaque caller-supplied record; no scientific interpretation is assigned here.
 pub observation_payload:Vec<u8>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalExperimentalObservation {
 pub observation_id:String, pub execution_id:String, pub observer_id:String, pub institution_id:String,
 pub observed_at:String, pub disposition:ObservationDisposition, pub commitment_event_id:String,
 pub challenge_id:String, pub criterion_id:String, pub criterion_generation:String,
 pub candidate_id:String, pub source_candidate_id:String, pub test_specification_id:String,
 pub measurement_specification_id:String, pub binding_digest:String, pub observation_digest:String,
 pub record_digest:String,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservationError { MissingField(&'static str), EmptyPayload, InvalidTimestamp, CommitmentVerification(VerificationFailure) }
impl fmt::Display for ObservationError {
 fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result { match self {
 Self::MissingField(x)=>write!(f,"missing required field: {x}"), Self::EmptyPayload=>write!(f,"observation payload must not be empty"),
 Self::InvalidTimestamp=>write!(f,"observed_at must be canonical UTC YYYY-MM-DDTHH:MM:SSZ"),
 Self::CommitmentVerification(e)=>write!(f,"commitment verification failed: {e}") } }
}
impl std::error::Error for ObservationError {}
impl ExternalExperimentalObservation {
 /// Ingest only against a verified prospective commitment and exact candidate binding.
 pub fn ingest(c:&CandidatePredictionCommitment,b:&CandidatePredictionBinding,i:ExternalObservationInput)->Result<Self,ObservationError>{
  for (n,v) in [("observation_id",&i.observation_id),("execution_id",&i.execution_id),("observer_id",&i.observer_id),("institution_id",&i.institution_id)] {
   if v.trim().is_empty(){return Err(ObservationError::MissingField(n));}
  }
  if i.observation_payload.is_empty(){return Err(ObservationError::EmptyPayload);}
  if !valid_utc(&i.observed_at){return Err(ObservationError::InvalidTimestamp);}
  c.verify_binding_detailed(b).map_err(ObservationError::CommitmentVerification)?;
  let mut o=Self{observation_id:i.observation_id,execution_id:i.execution_id,observer_id:i.observer_id,institution_id:i.institution_id,
   observed_at:i.observed_at,disposition:i.disposition,commitment_event_id:c.commitment.event_id().into(),
   challenge_id:c.commitment.challenge_id().into(),criterion_id:c.commitment.mapping_generation().into(),criterion_generation:c.commitment.criteria_generation().into(),
   candidate_id:c.candidate_id.clone(),source_candidate_id:c.source_candidate_id.clone(),test_specification_id:c.test_specification_id.clone(),
   measurement_specification_id:c.measurement_specification_id.clone(),binding_digest:c.binding_digest.clone(),
   observation_digest:hash(&i.observation_payload),record_digest:String::new()};
  o.record_digest=o.digest(); Ok(o)
 }
 /// Verify envelope integrity only; this is not scientific validation.
 pub fn verify_integrity(&self)->bool {
  [self.observation_id.as_str(),self.execution_id.as_str(),self.observer_id.as_str(),self.institution_id.as_str(),
   self.commitment_event_id.as_str(),self.challenge_id.as_str(),self.criterion_id.as_str(),self.criterion_generation.as_str(),self.candidate_id.as_str(),self.source_candidate_id.as_str(),self.test_specification_id.as_str(),
   self.measurement_specification_id.as_str(),self.binding_digest.as_str(),self.observation_digest.as_str()].iter().all(|s|!s.trim().is_empty())
   && valid_utc(&self.observed_at) && self.record_digest==self.digest()
 }
 fn digest(&self)->String{
  let mut h=Sha256::new(); h.update(b"symthaea:external-experimental-observation:v2\0");
  for s in [self.observation_id.as_str(),self.execution_id.as_str(),self.observer_id.as_str(),self.institution_id.as_str(),
   self.observed_at.as_str(),code(self.disposition),self.commitment_event_id.as_str(),self.challenge_id.as_str(),self.criterion_id.as_str(),self.criterion_generation.as_str(),self.candidate_id.as_str(),
   self.source_candidate_id.as_str(),self.test_specification_id.as_str(),self.measurement_specification_id.as_str(),
   self.binding_digest.as_str(),self.observation_digest.as_str()] { h.update((s.len() as u64).to_be_bytes()); h.update(s.as_bytes()); }
  format!("sha256:{:x}",h.finalize())
 }
}
fn code(d:ObservationDisposition)->&'static str{match d{ObservationDisposition::Reported=>"reported",ObservationDisposition::Failed=>"failed",
 ObservationDisposition::NullResult=>"null_result",ObservationDisposition::Contradictory=>"contradictory",ObservationDisposition::Inconclusive=>"inconclusive"}}
fn hash(b:&[u8])->String{let mut h=Sha256::new();h.update(b);format!("sha256:{:x}",h.finalize())}
fn valid_utc(s:&str)->bool{
 let b=s.as_bytes(); if b.len()!=20||b[4]!=b'-'||b[7]!=b'-'||b[10]!=b'T'||b[13]!=b':'||b[16]!=b':'||b[19]!=b'Z'
 ||![0,1,2,3,5,6,8,9,11,12,14,15,17,18].iter().all(|&i|b[i].is_ascii_digit()){return false;}
 let pair=|i:usize|((b[i]-b'0') as u32)*10+(b[i+1]-b'0') as u32;
 let y=(b[0]-b'0') as u32*1000+(b[1]-b'0') as u32*100+(b[2]-b'0') as u32*10+(b[3]-b'0') as u32;
 let m=pair(5);let d=pair(8);if !(1..=12).contains(&m)||pair(11)>23||pair(14)>59||pair(17)>59{return false;}
 let leap=y%4==0&&(y%100!=0||y%400==0);let max=match m{1|3|5|7|8|10|12=>31,4|6|9|11=>30,2 if leap=>29,2=>28,_=>return false};d>=1&&d<=max
}
#[cfg(test)] mod tests{
 use super::*;use crate::candidate_commitment::{CandidatePredictionSource,CandidatePredictionBinding,commit_candidate_envelope};use crate::prospective::ProspectiveProvenance;
 fn source()->CandidatePredictionSource{CandidatePredictionSource{candidate_id:"candidate:1".into(),source_candidate_id:"source:1".into(),test_specification_id:"test:1".into(),measurement_specification_id:"measure:1".into(),left_lineage:"left".into(),right_lineage:"right".into()}}
 fn setup()->(CandidatePredictionCommitment,CandidatePredictionBinding){let s=source();let b=CandidatePredictionBinding::from_source(&s,b"prediction").unwrap();let p=ProspectiveProvenance::new("input","artifact",b.lineage_digest().unwrap(),"2026-09-28T08:00:00Z","2026-09-28T09:00:00Z").unwrap();(commit_candidate_envelope(&s,"challenge","criteria","mapping","predictor","2026-09-28T09:00:00Z",p,b"prediction").unwrap(),b)}
 fn input(d:ObservationDisposition)->ExternalObservationInput{ExternalObservationInput{observation_id:"obs:1".into(),execution_id:"exec:1".into(),observer_id:"observer:1".into(),institution_id:"institution:1".into(),observed_at:"2026-09-29T10:00:00Z".into(),disposition:d,observation_payload:b"opaque record".to_vec()}}
 #[test]fn verified_binding_ingests_and_roundtrips(){let(c,b)=setup();let o=ExternalExperimentalObservation::ingest(&c,&b,input(ObservationDisposition::Reported)).unwrap();assert!(o.verify_integrity());assert_eq!(o.challenge_id,"challenge");assert_eq!(o.criterion_id,"mapping");assert_eq!(o.criterion_generation,"criteria");let j=serde_json::to_vec(&o).unwrap();assert_eq!(serde_json::from_slice::<ExternalExperimentalObservation>(&j).unwrap(),o);}
 #[test]fn preserves_negative_null_contradictory_and_inconclusive(){let(c,b)=setup();for d in [ObservationDisposition::Failed,ObservationDisposition::NullResult,ObservationDisposition::Contradictory,ObservationDisposition::Inconclusive]{let o=ExternalExperimentalObservation::ingest(&c,&b,input(d)).unwrap();assert_eq!(o.disposition,d);}}
 #[test]fn rejects_binding_mismatch(){let(c,mut b)=setup();b.candidate_id="changed".into();assert!(matches!(ExternalExperimentalObservation::ingest(&c,&b,input(ObservationDisposition::Reported)),Err(ObservationError::CommitmentVerification(_))));}
 #[test]fn rejects_missing_institution_empty_payload_and_invalid_date(){let(c,b)=setup();let mut i=input(ObservationDisposition::Reported);i.institution_id=" ".into();assert!(matches!(ExternalExperimentalObservation::ingest(&c,&b,i),Err(ObservationError::MissingField("institution_id"))));let mut i=input(ObservationDisposition::Reported);i.observation_payload.clear();assert_eq!(ExternalExperimentalObservation::ingest(&c,&b,i),Err(ObservationError::EmptyPayload));let mut i=input(ObservationDisposition::Reported);i.observed_at="2026-02-29T10:00:00Z".into();assert_eq!(ExternalExperimentalObservation::ingest(&c,&b,i),Err(ObservationError::InvalidTimestamp));}
 #[test]fn detects_record_tampering(){let(c,b)=setup();let mut o=ExternalExperimentalObservation::ingest(&c,&b,input(ObservationDisposition::Reported)).unwrap();o.observer_id="changed".into();assert!(!o.verify_integrity());}
}