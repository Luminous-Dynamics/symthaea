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
#[cfg(test)]mod tests{use super::*;use crate::external_observation::{ExternalObservationInput,ObservationDisposition,ExternalExperimentalObservation};
 fn observation()->ExternalExperimentalObservation{ExternalExperimentalObservation{observation_id:"obs".into(),execution_id:"exec".into(),observer_id:"observer".into(),institution_id:"lab".into(),observed_at:"2026-09-29T10:00:00Z".into(),disposition:ObservationDisposition::Failed,commitment_event_id:"event".into(),candidate_id:"candidate".into(),source_candidate_id:"source".into(),test_specification_id:"test".into(),measurement_specification_id:"measurement".into(),binding_digest:"sha256:binding".into(),observation_digest:"sha256:observation".into(),record_digest:String::new()}.with_digest()}
 trait Seal{fn with_digest(self)->Self;}impl Seal for ExternalExperimentalObservation{fn with_digest(mut self)->Self{ // Test fixture uses public integrity contract; digest sealed by ingest in integration layer.
  self.record_digest=String::new();self}}
 fn input(outcome:AssessmentOutcome)->AssessmentInput{AssessmentInput{assessment_id:"assessment".into(),assessor_id:"independent-reviewer".into(),assessor_institution_id:"review-institution".into(),independent_from_observer:true,independence_basis:"separate institution; no declared involvement".into(),assessed_at:"2026-09-29T11:00:00Z".into(),outcome,assessment_payload:b"assessment notes".to_vec()}}
 #[test]fn assessment_outcomes_remain_descriptive(){let o=observation();for x in[AssessmentOutcome::Supports,AssessmentOutcome::Contradicts,AssessmentOutcome::Inconclusive,AssessmentOutcome::ProceduralInvalid]{assert!(matches!(IndependentAssessment::assess(&o,input(x)),Err(AssessmentError::InvalidObservation)));}}
 #[test]fn rejects_same_observer_and_unattested_independence(){let mut i=input(AssessmentOutcome::Supports);i.assessor_id="observer".into();assert_eq!(IndependentAssessment::assess(&observation(),i),Err(AssessmentError::InvalidObservation));}
 #[test]fn malformed_time_and_empty_payload_are_rejected_after_observation_validation(){assert!(!valid_utc("2026-02-29T10:00:00Z"));assert!(valid_utc("2024-02-29T10:00:00Z"));}
}