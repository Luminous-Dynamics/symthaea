// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent-assessment claim envelopes over externally ingested observations.
//! These records do not adjudicate truth or promote replication/criterion evidence.
use serde::{Deserialize,Serialize}; use sha2::{Digest,Sha256}; use std::fmt;
use crate::external_observation::ExternalExperimentalObservation;

#[derive(Debug,Clone,Copy,PartialEq,Eq,Serialize,Deserialize)]
#[serde(rename_all="snake_case")]
pub enum AssessmentOutcome { Supports, Contradicts, Inconclusive, ProceduralInvalid }
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize)]
pub struct AssessmentInput {
 pub assessment_id:String,pub assessor_id:String,pub assessor_institution_id:String,
 /// Explicit assessor attestation; not independently authenticated by this crate.
 pub independent_from_observer:bool,pub independence_basis:String,pub assessed_at:String,
 pub outcome:AssessmentOutcome,pub assessment_payload:Vec<u8>,
}
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize)]
pub struct IndependentAssessment {
 pub assessment_id:String,pub assessor_id:String,pub assessor_institution_id:String,
 pub independent_from_observer:bool,pub independence_basis:String,pub assessed_at:String,
 pub outcome:AssessmentOutcome,pub observation_id:String,pub observation_record_digest:String,
 pub commitment_event_id:String,pub candidate_id:String,pub binding_digest:String,
 pub assessment_digest:String,pub record_digest:String,
}
#[derive(Debug,Clone,PartialEq,Eq)]
pub enum AssessmentError { MissingField(&'static str), EmptyPayload, NotIndependent, InvalidTimestamp, InvalidObservation }
impl fmt::Display for AssessmentError {fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{match self{
 Self::MissingField(s)=>write!(f,"missing required field: {s}"),Self::EmptyPayload=>write!(f,"assessment payload must not be empty"),
 Self::NotIndependent=>write!(f,"independence must be explicitly attested"),Self::InvalidTimestamp=>write!(f,"assessed_at must be canonical UTC YYYY-MM-DDTHH:MM:SSZ"),
 Self::InvalidObservation=>write!(f,"observation envelope integrity check failed")}}}
impl std::error::Error for AssessmentError{}
impl IndependentAssessment{
 /// Record a separate assessor's explicit assessment of one exact observation.
 pub fn assess(o:&ExternalExperimentalObservation,i:AssessmentInput)->Result<Self,AssessmentError>{
  if !o.verify_integrity(){return Err(AssessmentError::InvalidObservation);}
  for(n,v)in[("assessment_id",&i.assessment_id),("assessor_id",&i.assessor_id),("assessor_institution_id",&i.assessor_institution_id),("independence_basis",&i.independence_basis)]{if v.trim().is_empty(){return Err(AssessmentError::MissingField(n));}}
  if !i.independent_from_observer || i.assessor_id==o.observer_id {return Err(AssessmentError::NotIndependent);}
  if !valid_utc(&i.assessed_at){return Err(AssessmentError::InvalidTimestamp);}
  if i.assessment_payload.is_empty(){return Err(AssessmentError::EmptyPayload);}
  let assessment_digest=hash(b"symthaea:independent-assessment-payload:v1\\0",&i.assessment_payload);
  let mut a=Self{assessment_id:i.assessment_id,assessor_id:i.assessor_id,assessor_institution_id:i.assessor_institution_id,
   independent_from_observer:i.independent_from_observer,independence_basis:i.independence_basis,assessed_at:i.assessed_at,
   outcome:i.outcome,observation_id:o.observation_id.clone(),observation_record_digest:o.record_digest.clone(),
   commitment_event_id:o.commitment_event_id.clone(),candidate_id:o.candidate_id.clone(),binding_digest:o.binding_digest.clone(),
   assessment_digest,record_digest:String::new()};a.record_digest=a.digest();Ok(a)
 }
 /// Checks envelope consistency only; it cannot establish assessor independence or truth.
 pub fn verify_integrity(&self)->bool{
  [self.assessment_id.as_str(),self.assessor_id.as_str(),self.assessor_institution_id.as_str(),self.independence_basis.as_str(),
   self.observation_id.as_str(),self.observation_record_digest.as_str(),self.commitment_event_id.as_str(),self.candidate_id.as_str(),
   self.binding_digest.as_str(),self.assessment_digest.as_str()].iter().all(|s|!s.trim().is_empty())
   && self.independent_from_observer && valid_utc(&self.assessed_at)&&self.record_digest==self.digest()
 }
 fn digest(&self)->String{let mut h=Sha256::new();h.update(b"symthaea:independent-assessment-record:v1\\0");
  for s in [self.assessment_id.as_str(),self.assessor_id.as_str(),self.assessor_institution_id.as_str(),
   if self.independent_from_observer{"true"}else{"false"},self.independence_basis.as_str(),self.assessed_at.as_str(),
   outcome_code(self.outcome),self.observation_id.as_str(),self.observation_record_digest.as_str(),self.commitment_event_id.as_str(),
   self.candidate_id.as_str(),self.binding_digest.as_str(),self.assessment_digest.as_str()] {h.update((s.len()as u64).to_be_bytes());h.update(s.as_bytes());}
  format!("sha256:{:x}",h.finalize())}
}
fn outcome_code(o:AssessmentOutcome)->&'static str{match o{AssessmentOutcome::Supports=>"supports",AssessmentOutcome::Contradicts=>"contradicts",AssessmentOutcome::Inconclusive=>"inconclusive",AssessmentOutcome::ProceduralInvalid=>"procedural_invalid"}}
fn hash(domain:&[u8],b:&[u8])->String{let mut h=Sha256::new();h.update(domain);h.update((b.len()as u64).to_be_bytes());h.update(b);format!("sha256:{:x}",h.finalize())}
fn valid_utc(s:&str)->bool{let b=s.as_bytes();if b.len()!=20||b[4]!=b'-'||b[7]!=b'-'||b[10]!=b'T'||b[13]!=b':'||b[16]!=b':'||b[19]!=b'Z'||![0,1,2,3,5,6,8,9,11,12,14,15,17,18].iter().all(|&i|b[i].is_ascii_digit()){return false;}
 let pair=|i:usize|((b[i]-b'0')as u32)*10+(b[i+1]-b'0')as u32;let y=(b[0]-b'0')as u32*1000+(b[1]-b'0')as u32*100+(b[2]-b'0')as u32*10+(b[3]-b'0')as u32;
 let m=pair(5);let d=pair(8);if !(1..=12).contains(&m)||pair(11)>23||pair(14)>59||pair(17)>59{return false;}let leap=y%4==0&&(y%100!=0||y%400==0);let max=match m{1|3|5|7|8|10|12=>31,4|6|9|11=>30,2 if leap=>29,2=>28,_=>return false};d>=1&&d<=max}
#[cfg(test)]mod tests{
 use super::*;use crate::candidate_commitment::{CandidatePredictionSource,CandidatePredictionBinding,commit_candidate_envelope};
 use crate::external_observation::{ExternalObservationInput,ObservationDisposition,ExternalExperimentalObservation};
 use crate::prospective::ProspectiveProvenance;
 fn observation()->ExternalExperimentalObservation{
  let src=CandidatePredictionSource{candidate_id:"candidate:1".into(),source_candidate_id:"source:1".into(),test_specification_id:"test:1".into(),measurement_specification_id:"measure:1".into(),left_lineage:"left".into(),right_lineage:"right".into()};
  let b=CandidatePredictionBinding::from_source(&src,b"prediction").unwrap();let lin=b.lineage_digest().unwrap();
  let p=ProspectiveProvenance::new("input","artifact",lin,"2026-09-28T08:00:00Z","2026-09-28T09:00:00Z").unwrap();
  let c=commit_candidate_envelope(&src,"challenge","criteria","mapping","predictor","2026-09-28T09:00:00Z",p,b"prediction").unwrap();
  ExternalExperimentalObservation::ingest(&c,&b,ExternalObservationInput{observation_id:"obs:1".into(),execution_id:"exec:1".into(),observer_id:"observer:1".into(),institution_id:"lab".into(),observed_at:"2026-09-29T10:00:00Z".into(),disposition:ObservationDisposition::Failed,observation_payload:b"opaque observation".to_vec()}).unwrap()
 }
 fn input(outcome:AssessmentOutcome)->AssessmentInput{AssessmentInput{assessment_id:"assessment:1".into(),assessor_id:"independent-reviewer".into(),assessor_institution_id:"review-institution".into(),independent_from_observer:true,independence_basis:"separate institutional review; no declared participation".into(),assessed_at:"2026-09-29T11:00:00Z".into(),outcome,assessment_payload:b"assessment notes".to_vec()}}
 #[test]fn creates_integrity_verifiable_assessment(){let o=observation();let a=IndependentAssessment::assess(&o,input(AssessmentOutcome::Contradicts)).unwrap();assert!(a.verify_integrity());assert_eq!(a.observation_record_digest,o.record_digest);let bytes=serde_json::to_vec(&a).unwrap();let decoded:IndependentAssessment=serde_json::from_slice(&bytes).unwrap();assert_eq!(a,decoded);assert!(decoded.verify_integrity());}
 #[test]fn preserves_support_negative_inconclusive_and_procedural_outcomes(){let o=observation();for x in[AssessmentOutcome::Supports,AssessmentOutcome::Contradicts,AssessmentOutcome::Inconclusive,AssessmentOutcome::ProceduralInvalid]{let a=IndependentAssessment::assess(&o,input(x)).unwrap();assert_eq!(a.outcome,x);}}
 #[test]fn rejects_same_observer_and_missing_independence_attestation(){let o=observation();let mut i=input(AssessmentOutcome::Supports);i.assessor_id=o.observer_id.clone();assert_eq!(IndependentAssessment::assess(&o,i),Err(AssessmentError::NotIndependent));let mut i=input(AssessmentOutcome::Supports);i.independent_from_observer=false;assert_eq!(IndependentAssessment::assess(&o,i),Err(AssessmentError::NotIndependent));}
 #[test]fn rejects_invalid_timestamp_empty_payload_and_tampered_observation(){let o=observation();let mut i=input(AssessmentOutcome::Inconclusive);i.assessed_at="2026-02-29T11:00:00Z".into();assert_eq!(IndependentAssessment::assess(&o,i),Err(AssessmentError::InvalidTimestamp));let mut i=input(AssessmentOutcome::Inconclusive);i.assessment_payload.clear();assert_eq!(IndependentAssessment::assess(&o,i),Err(AssessmentError::EmptyPayload));let mut bad=o;bad.observer_id="changed".into();assert_eq!(IndependentAssessment::assess(&bad,input(AssessmentOutcome::Supports)),Err(AssessmentError::InvalidObservation));}
 #[test]fn detects_assessment_record_tampering(){let o=observation();let mut a=IndependentAssessment::assess(&o,input(AssessmentOutcome::Supports)).unwrap();a.assessor_institution_id="changed".into();assert!(!a.verify_integrity());}
}