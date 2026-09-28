// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Immutable prospective prediction commitments. This module cannot emit
//! observations, replications, or official-criterion evidence.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProspectiveProvenance {
    pub exact_input_digest: String,
    pub artifact_digest: String,
    pub model_lineage: String,
    pub knowledge_cutoff: String,
    pub exposure_cutoff: String,
}
impl ProspectiveProvenance {
    pub fn new(input: impl Into<String>, artifact: impl Into<String>, lineage: impl Into<String>, knowledge: impl Into<String>, exposure: impl Into<String>) -> Result<Self, CommitmentError> {
        let p=Self { exact_input_digest:input.into(), artifact_digest:artifact.into(), model_lineage:lineage.into(), knowledge_cutoff:knowledge.into(), exposure_cutoff:exposure.into() };
        for (n,v) in [("exact_input_digest",&p.exact_input_digest),("artifact_digest",&p.artifact_digest),("model_lineage",&p.model_lineage),("knowledge_cutoff",&p.knowledge_cutoff),("exposure_cutoff",&p.exposure_cutoff)] { if v.trim().is_empty(){return Err(CommitmentError::MissingField(n));} }
        validate_cutoff(&p.knowledge_cutoff)?;
        validate_cutoff(&p.exposure_cutoff)?;
        if p.exposure_cutoff < p.knowledge_cutoff { return Err(CommitmentError::ExposureBeforeKnowledgeCutoff); }
        Ok(p)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProspectivePredictionCommitment {
    event_id:String, challenge_id:String, criteria_generation:String, mapping_generation:String,
    actor_id:String, created_at:String, provenance:ProspectiveProvenance, payload_digest:String, parent_event_ids:Vec<String>,
}
impl ProspectivePredictionCommitment {
    pub fn commit(challenge:impl Into<String>, criteria:impl Into<String>, mapping:impl Into<String>, actor:impl Into<String>, created:impl Into<String>, provenance:ProspectiveProvenance, payload:&[u8])->Result<Self,CommitmentError>{
        let (challenge_id,criteria_generation,mapping_generation,actor_id,created_at)=(challenge.into(),criteria.into(),mapping.into(),actor.into(),created.into());
        for (n,v) in [("challenge_id",&challenge_id),("criteria_generation",&criteria_generation),("mapping_generation",&mapping_generation),("actor_id",&actor_id),("created_at",&created_at)] {if v.trim().is_empty(){return Err(CommitmentError::MissingField(n));}}
        if payload.is_empty(){return Err(CommitmentError::EmptyPredictionPayload);}
        let payload_digest=hash(payload);
        let mut c=Self{event_id:String::new(),challenge_id,criteria_generation,mapping_generation,actor_id,created_at,provenance,payload_digest,parent_event_ids:vec![]};
        c.event_id=c.digest(); Ok(c)
    }
    fn digest(&self)->String{
        let mut h=Sha256::new(); h.update(b"symthaea:prospective-prediction-commitment:v1\0");
        for v in [&self.challenge_id,&self.criteria_generation,&self.mapping_generation,&self.actor_id,&self.created_at,&self.provenance.exact_input_digest,&self.provenance.artifact_digest,&self.provenance.model_lineage,&self.provenance.knowledge_cutoff,&self.provenance.exposure_cutoff,&self.payload_digest] {h.update((v.len() as u64).to_be_bytes());h.update(v.as_bytes());}
        h.update((self.parent_event_ids.len() as u64).to_be_bytes());for p in &self.parent_event_ids{h.update((p.len() as u64).to_be_bytes());h.update(p.as_bytes());} format!("sha256:{:x}",h.finalize())
    }
    pub fn event_id(&self)->&str{&self.event_id}
    pub fn payload_digest(&self)->&str{&self.payload_digest}
    pub fn provenance(&self)->&ProspectiveProvenance{&self.provenance}
    pub fn parent_event_ids(&self)->&[String]{&self.parent_event_ids}
    pub fn supersede(&self,actor:impl Into<String>,created:impl Into<String>,payload:&[u8],provenance:ProspectiveProvenance)->Result<Self,CommitmentError>{
        let mut n=Self::commit(self.challenge_id.clone(),self.criteria_generation.clone(),self.mapping_generation.clone(),actor,created,provenance,payload)?;
        n.parent_event_ids.push(self.event_id.clone());n.event_id=n.digest();Ok(n)
    }
}
fn validate_cutoff(value:&str)->Result<(),CommitmentError>{
    let b=value.as_bytes();
    if b.len()!=20 || b[4]!=b'-' || b[7]!=b'-' || b[10]!=b'T' || b[13]!=b':' || b[16]!=b':' || b[19]!=b'Z'
        || ![0,1,2,3,5,6,8,9,11,12,14,15,17,18].iter().all(|&i| b[i].is_ascii_digit()) {
        return Err(CommitmentError::InvalidCutoff);
    }
    Ok(())
}
fn hash(bytes:&[u8])->String{let mut h=Sha256::new();h.update(bytes);format!("sha256:{:x}",h.finalize())}
#[derive(Debug,Clone,PartialEq,Eq)]
pub enum CommitmentError{MissingField(&'static str),EmptyPredictionPayload,ExposureBeforeKnowledgeCutoff,InvalidCutoff}
impl fmt::Display for CommitmentError{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{write!(f,"{self:?}")}}
impl std::error::Error for CommitmentError{}
