// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Eligibility envelopes for externally adjudicated official criterion evidence.
//! This module records an authority's explicit eligibility disposition; it never
//! decides scientific truth or automatically completes a Millennium criterion.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

use crate::external_observation::ExternalExperimentalObservation;
use crate::independent_assessment::IndependentAssessment;
use crate::replication::ReplicationRecord;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CriterionEvidenceDisposition {
    Eligible,
    Ineligible,
    Deferred,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CriterionEvidenceInput {
    pub evidence_id: String,
    pub challenge_id: String,
    pub criterion_id: String,
    pub criterion_generation: String,
    pub authority_id: String,
    pub authority_institution_id: String,
    /// Supplied attestation only; this crate does not authenticate authority.
    pub authority_basis: String,
    pub adjudicated_at: String,
    pub disposition: CriterionEvidenceDisposition,
    /// Opaque authority/scorer record. No scientific interpretation is assigned here.
    pub authority_payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CriterionEvidenceEligibility {
    pub evidence_id: String,
    pub challenge_id: String,
    pub criterion_id: String,
    pub criterion_generation: String,
    pub authority_id: String,
    pub authority_institution_id: String,
    pub authority_basis: String,
    pub adjudicated_at: String,
    pub disposition: CriterionEvidenceDisposition,
    pub observation_id: String,
    pub observation_record_digest: String,
    pub assessment_id: String,
    pub assessment_record_digest: String,
    pub replication_id: String,
    pub replication_record_digest: String,
    pub commitment_event_id: String,
    pub candidate_id: String,
    pub binding_digest: String,
    pub authority_payload_digest: String,
    pub record_digest: String,
    /// If present, this record supersedes the referenced immutable disposition.
    pub supersedes_evidence_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CriterionEvidenceError {
    MissingField(&'static str),
    EmptyPayload,
    InvalidTimestamp,
    InvalidObservation,
    InvalidAssessment,
    InvalidReplication,
    LinkMismatch(&'static str),
}

impl fmt::Display for CriterionEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(s) => write!(f, "missing required field: {s}"),
            Self::EmptyPayload => write!(f, "authority payload must not be empty"),
            Self::InvalidTimestamp => write!(f, "adjudicated_at must be canonical UTC YYYY-MM-DDTHH:MM:SSZ"),
            Self::InvalidObservation => write!(f, "observation envelope integrity check failed"),
            Self::InvalidAssessment => write!(f, "assessment envelope integrity check failed"),
            Self::InvalidReplication => write!(f, "replication envelope integrity check failed"),
            Self::LinkMismatch(s) => write!(f, "criterion-evidence linkage mismatch: {s}"),
        }
    }
}
impl std::error::Error for CriterionEvidenceError {}

impl CriterionEvidenceEligibility {
    /// Records an explicit external eligibility disposition over the complete evidence chain.
    /// The disposition is supplied by the authority; it is not inferred by Symthaea.
    pub fn adjudicate(
        observation: &ExternalExperimentalObservation,
        assessment: &IndependentAssessment,
        replication: &ReplicationRecord,
        input: CriterionEvidenceInput,
    ) -> Result<Self, CriterionEvidenceError> {
        if !observation.verify_integrity() { return Err(CriterionEvidenceError::InvalidObservation); }
        if !assessment.verify_integrity() { return Err(CriterionEvidenceError::InvalidAssessment); }
        if !replication.verify_integrity() { return Err(CriterionEvidenceError::InvalidReplication); }

        if assessment.observation_id != observation.observation_id
            || assessment.observation_record_digest != observation.record_digest
            || assessment.commitment_event_id != observation.commitment_event_id
            || assessment.candidate_id != observation.candidate_id
            || assessment.binding_digest != observation.binding_digest {
            return Err(CriterionEvidenceError::LinkMismatch("assessment ancestry"));
        }
        if replication.observation_id != observation.observation_id
            || replication.observation_record_digest != observation.record_digest
            || replication.assessment_id != assessment.assessment_id
            || replication.assessment_record_digest != assessment.record_digest
            || replication.commitment_event_id != observation.commitment_event_id
            || replication.candidate_id != observation.candidate_id
            || replication.binding_digest != observation.binding_digest {
            return Err(CriterionEvidenceError::LinkMismatch("replication ancestry"));
        }

        for (name, value) in [
            ("evidence_id", &input.evidence_id),
            ("challenge_id", &input.challenge_id),
            ("criterion_id", &input.criterion_id),
            ("criterion_generation", &input.criterion_generation),
            ("authority_id", &input.authority_id),
            ("authority_institution_id", &input.authority_institution_id),
            ("authority_basis", &input.authority_basis),
        ] {
            if value.trim().is_empty() { return Err(CriterionEvidenceError::MissingField(name)); }
        }
        if !valid_utc(&input.adjudicated_at) { return Err(CriterionEvidenceError::InvalidTimestamp); }
        if input.authority_payload.is_empty() { return Err(CriterionEvidenceError::EmptyPayload); }
        if input.challenge_id != observation.challenge_id { return Err(CriterionEvidenceError::LinkMismatch("challenge_id")); }
        if input.criterion_id != observation.criterion_id { return Err(CriterionEvidenceError::LinkMismatch("criterion_id")); }
        if input.criterion_generation != observation.criterion_generation { return Err(CriterionEvidenceError::LinkMismatch("criterion_generation")); }

        let authority_payload_digest = hash(
            b"symthaea:criterion-evidence-authority-payload:v1\0",
            &input.authority_payload,
        );
        let mut r = Self {
            evidence_id: input.evidence_id,
            challenge_id: input.challenge_id,
            criterion_id: input.criterion_id,
            criterion_generation: input.criterion_generation,
            authority_id: input.authority_id,
            authority_institution_id: input.authority_institution_id,
            authority_basis: input.authority_basis,
            adjudicated_at: input.adjudicated_at,
            disposition: input.disposition,
            supersedes_evidence_id: None,
            observation_id: observation.observation_id.clone(),
            observation_record_digest: observation.record_digest.clone(),
            assessment_id: assessment.assessment_id.clone(),
            assessment_record_digest: assessment.record_digest.clone(),
            replication_id: replication.replication_id.clone(),
            replication_record_digest: replication.record_digest.clone(),
            commitment_event_id: observation.commitment_event_id.clone(),
            candidate_id: observation.candidate_id.clone(),
            binding_digest: observation.binding_digest.clone(),
            authority_payload_digest,
            record_digest: String::new(),
        };
        r.record_digest = r.digest();
        Ok(r)
    }

    /// Create a new immutable disposition; the parent record is never mutated.
    pub fn supersede(&self, input: CriterionEvidenceInput, observation: &ExternalExperimentalObservation, assessment: &IndependentAssessment, replication: &ReplicationRecord) -> Result<Self, CriterionEvidenceError> {
        if !self.verify_integrity() { return Err(CriterionEvidenceError::LinkMismatch("superseded record integrity")); }
        if input.evidence_id.trim().is_empty() { return Err(CriterionEvidenceError::MissingField("evidence_id")); }
        if input.evidence_id == self.evidence_id { return Err(CriterionEvidenceError::LinkMismatch("supersession evidence_id")); }
        let mut next = Self::adjudicate(observation, assessment, replication, input)?;
        next.supersedes_evidence_id = Some(self.evidence_id.clone());
        next.record_digest = next.digest();
        Ok(next)
    }

    /// Envelope integrity only. Eligible != completed, and this method does not authenticate authority.
    pub fn verify_integrity(&self) -> bool {
        [
            self.evidence_id.as_str(), self.challenge_id.as_str(), self.criterion_id.as_str(),
            self.criterion_generation.as_str(), self.authority_id.as_str(),
            self.authority_institution_id.as_str(), self.authority_basis.as_str(),
            self.observation_id.as_str(), self.observation_record_digest.as_str(),
            self.assessment_id.as_str(), self.assessment_record_digest.as_str(),
            self.replication_id.as_str(), self.replication_record_digest.as_str(),
            self.commitment_event_id.as_str(), self.candidate_id.as_str(),
            self.binding_digest.as_str(), self.authority_payload_digest.as_str(), self.supersedes_evidence_id.as_deref().unwrap_or(""),
        ].iter().all(|s| !s.trim().is_empty())
            && valid_utc(&self.adjudicated_at)
            && self.record_digest == self.digest()
    }

    fn digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"symthaea:criterion-evidence-eligibility:v2\0");
        for s in [
            self.evidence_id.as_str(), self.challenge_id.as_str(), self.criterion_id.as_str(),
            self.criterion_generation.as_str(), self.authority_id.as_str(),
            self.authority_institution_id.as_str(), self.authority_basis.as_str(),
            self.adjudicated_at.as_str(), disposition_code(self.disposition),
            self.observation_id.as_str(), self.observation_record_digest.as_str(),
            self.assessment_id.as_str(), self.assessment_record_digest.as_str(),
            self.replication_id.as_str(), self.replication_record_digest.as_str(),
            self.commitment_event_id.as_str(), self.candidate_id.as_str(), self.binding_digest.as_str(),
            self.authority_payload_digest.as_str(), self.supersedes_evidence_id.as_deref().unwrap_or(""),
        ] {
            h.update((s.len() as u64).to_be_bytes());
            h.update(s.as_bytes());
        }
        format!("sha256:{:x}", h.finalize())
    }
}

fn disposition_code(d: CriterionEvidenceDisposition) -> &'static str {
    match d {
        CriterionEvidenceDisposition::Eligible => "eligible",
        CriterionEvidenceDisposition::Ineligible => "ineligible",
        CriterionEvidenceDisposition::Deferred => "deferred",
    }
}
fn hash(domain: &[u8], bytes: &[u8]) -> String {
    let mut h = Sha256::new(); h.update(domain); h.update((bytes.len() as u64).to_be_bytes()); h.update(bytes);
    format!("sha256:{:x}", h.finalize())
}
fn valid_utc(s: &str) -> bool {
    let b=s.as_bytes();
    if b.len()!=20 || b[4]!=b'-'||b[7]!=b'-'||b[10]!=b'T'||b[13]!=b':'||b[16]!=b':'||b[19]!=b'Z'
      || ![0,1,2,3,5,6,8,9,11,12,14,15,17,18].iter().all(|&i| b[i].is_ascii_digit()) { return false; }
    let pair=|i:usize| (b[i]-b'0') as u32*10+(b[i+1]-b'0') as u32;
    let y=(b[0]-b'0') as u32*1000+(b[1]-b'0') as u32*100+(b[2]-b'0') as u32*10+(b[3]-b'0') as u32;
    let m=pair(5); let d=pair(8);
    if !(1..=12).contains(&m)||pair(11)>23||pair(14)>59||pair(17)>59{return false;}
    let leap=y%4==0&&(y%100!=0||y%400==0);
    let max=match m{1|3|5|7|8|10|12=>31,4|6|9|11=>30,2 if leap=>29,2=>28,_=>return false};
    d>=1&&d<=max
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_commitment::{commit_candidate_envelope,CandidatePredictionBinding,CandidatePredictionSource};
    use crate::external_observation::{ExternalObservationInput,ObservationDisposition};
    use crate::independent_assessment::{AssessmentInput,AssessmentOutcome};
    use crate::prospective::ProspectiveProvenance;

    fn chain() -> (ExternalExperimentalObservation,IndependentAssessment,ReplicationRecord) {
        let s=CandidatePredictionSource{candidate_id:"candidate:1".into(),source_candidate_id:"source:1".into(),test_specification_id:"test:1".into(),measurement_specification_id:"measure:1".into(),left_lineage:"left".into(),right_lineage:"right".into()};
        let b=CandidatePredictionBinding::from_source(&s,b"prediction").unwrap();
        let p=ProspectiveProvenance::new("input","artifact",b.lineage_digest().unwrap(),"2026-09-28T08:00:00Z","2026-09-28T09:00:00Z").unwrap();
        let c=commit_candidate_envelope(&s,"challenge-1","criterion-1","generation-1","predictor","2026-09-28T09:00:00Z",p,b"prediction").unwrap();
        let o=ExternalExperimentalObservation::ingest(&c,&b,ExternalObservationInput{observation_id:"obs:1".into(),execution_id:"exec:original".into(),observer_id:"observer:original".into(),institution_id:"institution:original".into(),observed_at:"2026-09-29T10:00:00Z".into(),disposition:ObservationDisposition::Reported,observation_payload:b"observation".to_vec()}).unwrap();
        let a=IndependentAssessment::assess(&o,AssessmentInput{assessment_id:"assessment:1".into(),assessor_id:"assessor:1".into(),assessor_institution_id:"review".into(),independent_from_observer:true,independence_basis:"separate".into(),assessed_at:"2026-09-29T11:00:00Z".into(),outcome:AssessmentOutcome::Supports,assessment_payload:b"assessment".to_vec()}).unwrap();
        let r=ReplicationRecord::record(&o,&a,crate::replication::ReplicationInput{replication_id:"replication:1".into(),execution_id:"exec:replica".into(),observer_id:"observer:replica".into(),institution_id:"institution:replica".into(),independent_from_original_observer:true,independence_basis:"distinct execution".into(),replicated_at:"2026-09-29T12:00:00Z".into(),outcome:crate::replication::ReplicationOutcome::ReplicatedSupportive,replication_payload:b"replication".to_vec()}).unwrap();
        (o,a,r)
    }
    fn input(d:CriterionEvidenceDisposition)->CriterionEvidenceInput{CriterionEvidenceInput{evidence_id:"evidence:1".into(),challenge_id:"challenge-1".into(),criterion_id:"criterion-1".into(),criterion_generation:"generation-1".into(),authority_id:"authority:1".into(),authority_institution_id:"institution:authority".into(),authority_basis:"official scorer designation".into(),adjudicated_at:"2026-09-29T13:00:00Z".into(),disposition:d,authority_payload:b"authority record".to_vec()}}
    #[test]fn binds_complete_chain(){let(o,a,r)=chain();let e=CriterionEvidenceEligibility::adjudicate(&o,&a,&r,input(CriterionEvidenceDisposition::Eligible)).unwrap();assert!(e.verify_integrity());assert_eq!(e.challenge_id,"challenge-1");assert_eq!(e.criterion_id,"generation-1");assert_eq!(e.criterion_generation,"criterion-1");assert_eq!(e.observation_record_digest,o.record_digest);assert_eq!(e.assessment_record_digest,a.record_digest);assert_eq!(e.replication_record_digest,r.record_digest);}
    #[test]fn preserves_explicit_dispositions(){let(o,a,r)=chain();for d in[CriterionEvidenceDisposition::Eligible,CriterionEvidenceDisposition::Ineligible,CriterionEvidenceDisposition::Deferred]{let e=CriterionEvidenceEligibility::adjudicate(&o,&a,&r,input(d)).unwrap();assert_eq!(e.disposition,d);}}
    #[test]fn rejects_wrong_chain(){let(o,mut a,r)=chain();a.candidate_id="changed".into();a.record_digest="stale".into();assert_eq!(CriterionEvidenceEligibility::adjudicate(&o,&a,&r,input(CriterionEvidenceDisposition::Deferred)),Err(CriterionEvidenceError::InvalidAssessment));}
    #[test]fn rejects_wrong_lineage_and_bad_input(){let(o,a,r)=chain();let mut i=input(CriterionEvidenceDisposition::Deferred);i.criterion_generation="other-generation".into();assert_eq!(CriterionEvidenceEligibility::adjudicate(&o,&a,&r,i),Err(CriterionEvidenceError::LinkMismatch("criterion_generation")));let mut i=input(CriterionEvidenceDisposition::Deferred);i.criterion_generation="".into();assert_eq!(CriterionEvidenceEligibility::adjudicate(&o,&a,&r,i),Err(CriterionEvidenceError::MissingField("criterion_generation")));let mut i=input(CriterionEvidenceDisposition::Deferred);i.adjudicated_at="2026-02-29T13:00:00Z".into();assert_eq!(CriterionEvidenceEligibility::adjudicate(&o,&a,&r,i),Err(CriterionEvidenceError::InvalidTimestamp));}
    #[test]fn supersession_is_new_immutable_record(){let(o,a,r)=chain();let original=CriterionEvidenceEligibility::adjudicate(&o,&a,&r,input(CriterionEvidenceDisposition::Deferred)).unwrap();let mut next=input(CriterionEvidenceDisposition::Eligible);next.evidence_id="evidence:2".into();let replacement=original.supersede(next,&o,&a,&r).unwrap();assert_eq!(replacement.supersedes_evidence_id.as_deref(),Some("evidence:1"));assert_ne!(replacement.record_digest,original.record_digest);assert!(original.verify_integrity());assert!(replacement.verify_integrity());}
    #[test]fn roundtrip_and_tamper_detection(){let(o,a,r)=chain();let e=CriterionEvidenceEligibility::adjudicate(&o,&a,&r,input(CriterionEvidenceDisposition::Deferred)).unwrap();let j=serde_json::to_vec(&e).unwrap();assert_eq!(serde_json::from_slice::<CriterionEvidenceEligibility>(&j).unwrap(),e);let mut t=e;t.disposition=CriterionEvidenceDisposition::Ineligible;assert!(!t.verify_integrity());}
}
