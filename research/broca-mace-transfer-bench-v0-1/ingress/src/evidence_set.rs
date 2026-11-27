// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic joint evidence-set governance.
//! Constraint authenticity is delegated to an injected verifier; local digests
//! are integrity commitments only, never signatures.

use std::collections::{BTreeMap, BTreeSet};
use sha2::{Digest, Sha256};
use super::{DependencyLineageStatusV1, EvidenceUseReceiptV1, Scope, SourceLifecycleStatus, SourceTrustStatus};

pub const EVIDENCE_SET_SCHEMA_V1: &str = "mace_evidence_set_selection_v1";
pub const EVIDENCE_SET_OBSTRUCTION_SCHEMA_V1: &str = "mace_evidence_set_obstruction_v1";
pub const MAX_EVIDENCE_CANDIDATES_V1: usize = 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidencePairRelationV1 { MustLink, CannotLink }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidencePairReasonV1 {
    Contradiction, Supersession, ScopeConflict, PrivacyConflict,
    DependencyRequires, PolicyRequires, DuplicateClaim,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidencePairConstraintV1 {
    pub constraint_id: String,
    pub left_evidence_id: String,
    pub right_evidence_id: String,
    pub relation: EvidencePairRelationV1,
    pub reason: EvidencePairReasonV1,
    pub issuer_id: String,
    pub authority_receipt_sha256: String,
    pub constraint_policy_sha256: String,
    pub canonical_sha256: String,
}

impl EvidencePairConstraintV1 {
    fn digest_without_commitment(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"luminous:mace:evidence-pair-constraint:v1\0");
        for value in [
            self.constraint_id.as_str(), self.left_evidence_id.as_str(),
            self.right_evidence_id.as_str(), self.issuer_id.as_str(),
            self.authority_receipt_sha256.as_str(), self.constraint_policy_sha256.as_str(),
        ] { put_text(&mut h, value); }
        h.update([
            match self.relation { EvidencePairRelationV1::MustLink => 1, EvidencePairRelationV1::CannotLink => 2 },
            match self.reason {
                EvidencePairReasonV1::Contradiction => 1,
                EvidencePairReasonV1::Supersession => 2,
                EvidencePairReasonV1::ScopeConflict => 3,
                EvidencePairReasonV1::PrivacyConflict => 4,
                EvidencePairReasonV1::DependencyRequires => 5,
                EvidencePairReasonV1::PolicyRequires => 6,
                EvidencePairReasonV1::DuplicateClaim => 7,
            },
        ]);
        hex(h.finalize().as_slice())
    }
    pub fn verify_integrity(&self) -> bool { self.canonical_sha256 == self.digest_without_commitment() }
    pub fn seal_integrity(&mut self) { self.canonical_sha256 = self.digest_without_commitment(); }
}

/// Production implementation must authenticate authority issuer, receipt,
/// consumer scope and constraint policy (for example via Mycelix).
pub trait EvidenceSetConstraintVerifierV1 {
    fn verify_constraint(&self, constraint: &EvidencePairConstraintV1, expected_policy_sha256: &str) -> bool;
}

#[derive(Clone, Debug)]
pub struct EvidenceSetContextV1 {
    pub valid_at_ms: i64,
    pub known_at_ms: i64,
    pub required_scope: Scope,
    pub purpose_code: String,
    pub consumer_policy_sha256: String,
    pub constraint_policy_sha256: String,
    pub require_complete_dependency_lineage: bool,
    pub max_token_cost: u64,
    pub max_items: usize,
    pub min_items: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct EvidenceSetCandidateV1<'a> {
    pub evidence_id: &'a str,
    pub representation_sha256: &'a str,
    pub token_cost: u64,
    pub relevance_score_micros: u64,
    pub required: bool,
    pub receipt: &'a EvidenceUseReceiptV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateExclusionReasonV1 {
    NotSelectedByOptimizer,
    MissingVerifiedIngressCapability,
    ReceiptIntegrityFailure,
    BitemporalSnapshotMismatch,
    ScopeMismatch,
    PurposeMismatch,
    ConsumerPolicyMismatch,
    SourceNotTrusted,
    SourceNotActive,
    DependencyLineageIncomplete,
    OutsideValidityInterval,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExcludedEvidenceV1 {
    pub evidence_id: String,
    pub reason: CandidateExclusionReasonV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceSetSelectionReceiptV1 {
    pub schema_version: String,
    pub context_sha256: String,
    pub candidate_set_sha256: String,
    pub constraint_set_sha256: String,
    pub constraint_policy_sha256: String,
    pub selected_evidence_ids: Vec<String>,
    pub selected_representation_sha256: Vec<String>,
    pub selected_use_receipt_sha256: Vec<String>,
    pub excluded: Vec<ExcludedEvidenceV1>,
    pub total_token_cost: u64,
    pub total_relevance_score_micros: u64,
    pub max_token_cost: u64,
    pub canonical_sha256: String,
}

impl EvidenceSetSelectionReceiptV1 {
    fn digest_without_commitment(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"luminous:mace:evidence-set-selection:v1\0");
        for value in [
            self.schema_version.as_str(), self.context_sha256.as_str(),
            self.candidate_set_sha256.as_str(), self.constraint_set_sha256.as_str(),
            self.constraint_policy_sha256.as_str(),
        ] { put_text(&mut h, value); }
        put_vec(&mut h, &self.selected_evidence_ids);
        put_vec(&mut h, &self.selected_representation_sha256);
        put_vec(&mut h, &self.selected_use_receipt_sha256);
        h.update((self.excluded.len() as u64).to_be_bytes());
        for excluded in &self.excluded {
            put_text(&mut h, &excluded.evidence_id);
            h.update([exclusion_code(excluded.reason)]);
        }
        h.update(self.total_token_cost.to_be_bytes());
        h.update(self.total_relevance_score_micros.to_be_bytes());
        h.update(self.max_token_cost.to_be_bytes());
        hex(h.finalize().as_slice())
    }
    pub fn verify_integrity(&self) -> bool { self.canonical_sha256 == self.digest_without_commitment() }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceSetObstructionReasonV1 {
    InvalidContext, TooManyCandidates, InvalidCandidateId, InvalidCandidateDigest,
    DuplicateEvidenceId, DuplicateEventId, InvalidConstraint,
    ConstraintVerificationFailed, UnknownConstraintEndpoint, DuplicateConstraintId,
    MustLinkCannotLinkConflict, RequiredConstraintConflict, RequiredSetExceedsBudget,
    NoFeasibleSelection,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceSetObstructionCertificateV1 {
    pub schema_version: String,
    pub reason: EvidenceSetObstructionReasonV1,
    pub context_sha256: String,
    pub candidate_set_sha256: String,
    pub constraint_set_sha256: String,
    pub blocking_evidence_ids: Vec<String>,
    pub blocking_constraint_ids: Vec<String>,
    pub canonical_sha256: String,
}

impl EvidenceSetObstructionCertificateV1 {
    fn digest_without_commitment(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"luminous:mace:evidence-set-obstruction:v1\0");
        for value in [self.schema_version.as_str(), self.context_sha256.as_str(),
            self.candidate_set_sha256.as_str(), self.constraint_set_sha256.as_str()] {
            put_text(&mut h, value);
        }
        h.update([obstruction_code(self.reason)]);
        put_vec(&mut h, &self.blocking_evidence_ids);
        put_vec(&mut h, &self.blocking_constraint_ids);
        hex(h.finalize().as_slice())
    }
    pub fn verify_integrity(&self) -> bool { self.canonical_sha256 == self.digest_without_commitment() }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CandidateState { Admissible, Excluded(CandidateExclusionReasonV1) }

fn hex(bytes: &[u8]) -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() }
fn put_text(h: &mut Sha256, value: &str) { h.update((value.len() as u64).to_be_bytes()); h.update(value.as_bytes()); }
fn put_vec(h: &mut Sha256, values: &[String]) { h.update((values.len() as u64).to_be_bytes()); for v in values { put_text(h, v); } }
fn valid_sha(v: &str) -> bool { v.len() == 64 && v.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) }
fn valid_id(v: &str) -> bool { !v.is_empty() && v.len() <= 128 && v.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.'|b'_'|b'-'|b':')) }

fn scope_code(scope: Scope) -> [u8; 2] {
    [
        match scope.regime { broca_mace_policy::Regime::Baseline => 1, broca_mace_policy::Regime::Shifted => 2 },
        match scope.objective { broca_mace_policy::Objective::Quality => 1, broca_mace_policy::Objective::Cost => 2, broca_mace_policy::Objective::Speed => 3 },
    ]
}
fn context_digest(c: &EvidenceSetContextV1) -> String {
    let mut h = Sha256::new();
    h.update(b"luminous:mace:evidence-set-context:v1\0");
    h.update(c.valid_at_ms.to_be_bytes()); h.update(c.known_at_ms.to_be_bytes()); h.update(scope_code(c.required_scope));
    put_text(&mut h, &c.purpose_code); put_text(&mut h, &c.consumer_policy_sha256);
    put_text(&mut h, &c.constraint_policy_sha256); h.update([u8::from(c.require_complete_dependency_lineage)]);
    h.update(c.max_token_cost.to_be_bytes()); h.update((c.max_items as u64).to_be_bytes()); h.update((c.min_items as u64).to_be_bytes());
    hex(h.finalize().as_slice())
}
fn candidate_digest(candidates: &[EvidenceSetCandidateV1<'_>]) -> String {
    let mut values: Vec<_> = candidates.iter().collect();
    values.sort_by(|a,b| a.evidence_id.cmp(b.evidence_id));
    let mut h = Sha256::new(); h.update(b"luminous:mace:evidence-set-candidates:v1\0");
    h.update((values.len() as u64).to_be_bytes());
    for c in values {
        put_text(&mut h,c.evidence_id); put_text(&mut h,c.representation_sha256);
        h.update(c.token_cost.to_be_bytes()); h.update(c.relevance_score_micros.to_be_bytes());
        h.update([u8::from(c.required)]); h.update(c.receipt.event_id.to_be_bytes());
        put_text(&mut h,&c.receipt.canonical_sha256);
    }
    hex(h.finalize().as_slice())
}
fn constraint_digest(constraints: &[EvidencePairConstraintV1]) -> String {
    let mut values: Vec<_> = constraints.iter().collect();
    values.sort_by(|a,b| a.constraint_id.cmp(&b.constraint_id));
    let mut h=Sha256::new(); h.update(b"luminous:mace:evidence-set-constraints:v1\0");
    h.update((values.len() as u64).to_be_bytes());
    for c in values { put_text(&mut h,&c.constraint_id); put_text(&mut h,&c.canonical_sha256); }
    hex(h.finalize().as_slice())
}
fn exclusion_code(v: CandidateExclusionReasonV1) -> u8 {
    match v {
        CandidateExclusionReasonV1::NotSelectedByOptimizer=>1,
        CandidateExclusionReasonV1::MissingVerifiedIngressCapability=>2,
        CandidateExclusionReasonV1::ReceiptIntegrityFailure=>3,
        CandidateExclusionReasonV1::BitemporalSnapshotMismatch=>4,
        CandidateExclusionReasonV1::ScopeMismatch=>5,
        CandidateExclusionReasonV1::PurposeMismatch=>6,
        CandidateExclusionReasonV1::ConsumerPolicyMismatch=>7,
        CandidateExclusionReasonV1::SourceNotTrusted=>8,
        CandidateExclusionReasonV1::SourceNotActive=>9,
        CandidateExclusionReasonV1::DependencyLineageIncomplete=>10,
        CandidateExclusionReasonV1::OutsideValidityInterval=>11,
    }
}
fn obstruction_code(v: EvidenceSetObstructionReasonV1) -> u8 {
    match v {
        EvidenceSetObstructionReasonV1::InvalidContext=>1,
        EvidenceSetObstructionReasonV1::TooManyCandidates=>2,
        EvidenceSetObstructionReasonV1::InvalidCandidateId=>3,
        EvidenceSetObstructionReasonV1::InvalidCandidateDigest=>4,
        EvidenceSetObstructionReasonV1::DuplicateEvidenceId=>5,
        EvidenceSetObstructionReasonV1::DuplicateEventId=>6,
        EvidenceSetObstructionReasonV1::InvalidConstraint=>7,
        EvidenceSetObstructionReasonV1::ConstraintVerificationFailed=>8,
        EvidenceSetObstructionReasonV1::UnknownConstraintEndpoint=>9,
        EvidenceSetObstructionReasonV1::DuplicateConstraintId=>10,
        EvidenceSetObstructionReasonV1::MustLinkCannotLinkConflict=>11,
        EvidenceSetObstructionReasonV1::RequiredConstraintConflict=>12,
        EvidenceSetObstructionReasonV1::RequiredSetExceedsBudget=>13,
        EvidenceSetObstructionReasonV1::NoFeasibleSelection=>14,
    }
}
fn obstruction(
    reason: EvidenceSetObstructionReasonV1, ctx: &str, cand: &str, cons: &str,
    mut ids: Vec<String>, mut constraint_ids: Vec<String>,
) -> EvidenceSetObstructionCertificateV1 {
    ids.sort(); ids.dedup(); constraint_ids.sort(); constraint_ids.dedup();
    let mut value=EvidenceSetObstructionCertificateV1 {
        schema_version:EVIDENCE_SET_OBSTRUCTION_SCHEMA_V1.into(), reason,
        context_sha256:ctx.into(),candidate_set_sha256:cand.into(),constraint_set_sha256:cons.into(),
        blocking_evidence_ids:ids,blocking_constraint_ids:constraint_ids,canonical_sha256:String::new()
    };
    value.canonical_sha256=value.digest_without_commitment(); value
}
fn receipt_exclusion(r:&EvidenceUseReceiptV1,c:&EvidenceSetContextV1)->Option<CandidateExclusionReasonV1>{
    if !r.has_verified_ingress_provenance(){return Some(CandidateExclusionReasonV1::MissingVerifiedIngressCapability);}
    if !r.verify_integrity(){return Some(CandidateExclusionReasonV1::ReceiptIntegrityFailure);}
    if r.valid_at_ms!=c.valid_at_ms || r.known_at_ms!=c.known_at_ms{return Some(CandidateExclusionReasonV1::BitemporalSnapshotMismatch);}
    if r.scope!=c.required_scope{return Some(CandidateExclusionReasonV1::ScopeMismatch);}
    if r.purpose_code!=c.purpose_code{return Some(CandidateExclusionReasonV1::PurposeMismatch);}
    if r.consumer_policy_sha256!=c.consumer_policy_sha256{return Some(CandidateExclusionReasonV1::ConsumerPolicyMismatch);}
    if r.source_trust_status!=SourceTrustStatus::TrustedForScope{return Some(CandidateExclusionReasonV1::SourceNotTrusted);}
    if r.lifecycle_status_as_known!=SourceLifecycleStatus::Active{return Some(CandidateExclusionReasonV1::SourceNotActive);}
    if c.require_complete_dependency_lineage && r.dependency_lineage_status!=DependencyLineageStatusV1::Complete {
        return Some(CandidateExclusionReasonV1::DependencyLineageIncomplete);
    }
    if c.valid_at_ms<r.valid_from_ms || r.valid_until_ms.is_some_and(|until|c.valid_at_ms>=until) {
        return Some(CandidateExclusionReasonV1::OutsideValidityInterval);
    }
    None
}
fn is_better(score:u64,cost:u64,ids:&[String],best_score:u64,best_cost:u64,best_ids:&[String])->bool {
    score>best_score || (score==best_score && cost<best_cost) || (score==best_score && cost==best_cost && ids<best_ids)
}

/// Exact maximum-utility selection with deterministic tie-breaking.
/// Up to 20 candidate items are supported; larger inputs fail closed.
pub fn select_evidence_set_v1(
    context:&EvidenceSetContextV1,
    candidates:&[EvidenceSetCandidateV1<'_>],
    constraints:&[EvidencePairConstraintV1],
    verifier:&impl EvidenceSetConstraintVerifierV1,
)->Result<EvidenceSetSelectionReceiptV1,EvidenceSetObstructionCertificateV1>{
    let ctx=context_digest(context); let cand=candidate_digest(candidates); let cons=constraint_digest(constraints);
    let fail=|reason,ids,cs|obstruction(reason,&ctx,&cand,&cons,ids,cs);
    if !valid_id(&context.purpose_code) || !valid_sha(&context.consumer_policy_sha256)
        || !valid_sha(&context.constraint_policy_sha256) || context.min_items>context.max_items {
        return Err(fail(EvidenceSetObstructionReasonV1::InvalidContext,vec![],vec![]));
    }
    if candidates.len()>MAX_EVIDENCE_CANDIDATES_V1 {
        return Err(fail(EvidenceSetObstructionReasonV1::TooManyCandidates,vec![],vec![]));
    }
    let mut indices=BTreeMap::new(); let mut events=BTreeSet::new(); let mut states=Vec::new();
    for (i,c) in candidates.iter().enumerate(){
        if !valid_id(c.evidence_id){return Err(fail(EvidenceSetObstructionReasonV1::InvalidCandidateId,vec![c.evidence_id.into()],vec![]));}
        if !valid_sha(c.representation_sha256){return Err(fail(EvidenceSetObstructionReasonV1::InvalidCandidateDigest,vec![c.evidence_id.into()],vec![]));}
        if indices.insert(c.evidence_id,i).is_some(){return Err(fail(EvidenceSetObstructionReasonV1::DuplicateEvidenceId,vec![c.evidence_id.into()],vec![]));}
        if !events.insert(c.receipt.event_id){return Err(fail(EvidenceSetObstructionReasonV1::DuplicateEventId,vec![c.evidence_id.into()],vec![]));}
        states.push(receipt_exclusion(c.receipt,context).map_or(CandidateState::Admissible,CandidateState::Excluded));
    }
    let mut seen_constraints=BTreeSet::new();
    for c in constraints {
        if !valid_id(&c.constraint_id)||!valid_id(&c.left_evidence_id)||!valid_id(&c.right_evidence_id)
            ||!valid_id(&c.issuer_id)||!valid_sha(&c.authority_receipt_sha256)
            ||c.constraint_policy_sha256!=context.constraint_policy_sha256||!c.verify_integrity(){
            return Err(fail(EvidenceSetObstructionReasonV1::InvalidConstraint,vec![c.left_evidence_id.clone(),c.right_evidence_id.clone()],vec![c.constraint_id.clone()]));
        }
        if !seen_constraints.insert(c.constraint_id.as_str()){
            return Err(fail(EvidenceSetObstructionReasonV1::DuplicateConstraintId,vec![],vec![c.constraint_id.clone()]));
        }
        if !verifier.verify_constraint(c,&context.constraint_policy_sha256){
            return Err(fail(EvidenceSetObstructionReasonV1::ConstraintVerificationFailed,vec![c.left_evidence_id.clone(),c.right_evidence_id.clone()],vec![c.constraint_id.clone()]));
        }
        if !indices.contains_key(c.left_evidence_id.as_str())||!indices.contains_key(c.right_evidence_id.as_str()){
            return Err(fail(EvidenceSetObstructionReasonV1::UnknownConstraintEndpoint,vec![c.left_evidence_id.clone(),c.right_evidence_id.clone()],vec![c.constraint_id.clone()]));
        }
    }

    let mut parent:Vec<usize>=(0..candidates.len()).collect();
    fn find(parent:&mut [usize],i:usize)->usize{
        if parent[i]!=i{let r=find(parent,parent[i]);parent[i]=r;} parent[i]
    }
    fn union(parent:&mut [usize],a:usize,b:usize){
        let x=find(parent,a);let y=find(parent,b);if x!=y{let (lo,hi)=if x<y{(x,y)}else{(y,x)};parent[hi]=lo;}
    }
    for c in constraints.iter().filter(|v|v.relation==EvidencePairRelationV1::MustLink){
        union(&mut parent,indices[c.left_evidence_id.as_str()],indices[c.right_evidence_id.as_str()]);
    }
    for c in constraints.iter().filter(|v|v.relation==EvidencePairRelationV1::CannotLink){
        let a=indices[c.left_evidence_id.as_str()];let b=indices[c.right_evidence_id.as_str()];
        if find(&mut parent,a)==find(&mut parent,b){
            let members=(0..candidates.len()).filter(|i|find(&mut parent,*i)==find(&mut parent,a))
                .map(|i|candidates[i].evidence_id.to_owned()).collect();
            let core=constraints.iter().filter(|v|v.relation==EvidencePairRelationV1::MustLink||v.constraint_id==c.constraint_id)
                .map(|v|v.constraint_id.clone()).collect();
            return Err(fail(EvidenceSetObstructionReasonV1::MustLinkCannotLinkConflict,members,core));
        }
    }
    let mut required_mask=0u64;
    for (i,c) in candidates.iter().enumerate(){if c.required{required_mask|=1u64<<i;}}
    let mut required_closure=required_mask;
    for i in 0..candidates.len(){
        if required_mask&(1u64<<i)!=0{
            let root=find(&mut parent,i);
            for j in 0..candidates.len(){if find(&mut parent,j)==root{required_closure|=1u64<<j;}}
        }
    }
    let mut required_cost=0u64;let mut required_count=0usize;let mut required_ids=Vec::new();
    for (i,c) in candidates.iter().enumerate(){
        if required_closure&(1u64<<i)!=0{
            required_ids.push(c.evidence_id.to_owned());
            if states[i]!=CandidateState::Admissible{
                return Err(fail(EvidenceSetObstructionReasonV1::RequiredConstraintConflict,required_ids,vec![]));
            }
            required_cost=required_cost.checked_add(c.token_cost).ok_or_else(||fail(EvidenceSetObstructionReasonV1::RequiredSetExceedsBudget,required_ids.clone(),vec![]))?;
            required_count+=1;
        }
    }
    let required_conflict:Vec<_>=constraints.iter().filter(|c|{
        c.relation==EvidencePairRelationV1::CannotLink
            && required_closure&(1u64<<indices[c.left_evidence_id.as_str()])!=0
            && required_closure&(1u64<<indices[c.right_evidence_id.as_str()])!=0
    }).map(|c|c.constraint_id.clone()).collect();
    if !required_conflict.is_empty(){
        return Err(fail(EvidenceSetObstructionReasonV1::RequiredConstraintConflict,required_ids,required_conflict));
    }
    if required_cost>context.max_token_cost||required_count>context.max_items{
        return Err(fail(EvidenceSetObstructionReasonV1::RequiredSetExceedsBudget,required_ids,vec![]));
    }

    let mut best:Option<(u64,u64,u64,Vec<String>)>=None;
    for mask in 0..(1u64<<candidates.len()){
        if mask&required_closure!=required_closure{continue;}
        let count=mask.count_ones() as usize;
        if count<context.min_items||count>context.max_items{continue;}
        let mut cost=0u64;let mut score=0u64;let mut ids=Vec::new();let mut valid=true;
        for (i,c) in candidates.iter().enumerate(){
            if mask&(1u64<<i)==0{continue;}
            if states[i]!=CandidateState::Admissible{valid=false;break;}
            let Some(next_cost)=cost.checked_add(c.token_cost) else {valid=false;break;};
            let Some(next_score)=score.checked_add(c.relevance_score_micros) else {valid=false;break;};
            cost=next_cost;score=next_score;ids.push(c.evidence_id.to_owned());
        }
        if !valid||cost>context.max_token_cost{continue;}
        for c in constraints{
            let left=mask&(1u64<<indices[c.left_evidence_id.as_str()])!=0;
            let right=mask&(1u64<<indices[c.right_evidence_id.as_str()])!=0;
            let satisfied=match c.relation{EvidencePairRelationV1::MustLink=>left==right,EvidencePairRelationV1::CannotLink=>!(left&&right)};
            if !satisfied{valid=false;break;}
        }
        if !valid{continue;}
        ids.sort();
        match &best{
            None=>best=Some((mask,score,cost,ids)),
            Some((_,bs,bc,bi)) if is_better(score,cost,&ids,*bs,*bc,bi)=>best=Some((mask,score,cost,ids)),
            _=>{}
        }
    }
    let Some((mask,score,cost,selected_ids))=best else{
        let invalid=candidates.iter().enumerate().filter(|(i,_)|states[*i]!=CandidateState::Admissible)
            .map(|(_,c)|c.evidence_id.to_owned()).collect();
        return Err(fail(EvidenceSetObstructionReasonV1::NoFeasibleSelection,invalid,constraints.iter().map(|c|c.constraint_id.clone()).collect()));
    };
    let mut selected_representations=Vec::new();let mut selected_receipts=Vec::new();let mut excluded=Vec::new();
    for (i,c) in candidates.iter().enumerate(){
        if mask&(1u64<<i)!=0{
            selected_representations.push((c.evidence_id,c.representation_sha256,c.receipt.canonical_sha256.as_str()));
        }else{
            let reason=match states[i]{CandidateState::Admissible=>CandidateExclusionReasonV1::NotSelectedByOptimizer,CandidateState::Excluded(r)=>r};
            excluded.push(ExcludedEvidenceV1{evidence_id:c.evidence_id.to_owned(),reason});
        }
    }
    selected_representations.sort_by(|a,b|a.0.cmp(b.0));
    for (_,representation,receipt) in selected_representations{
        selected_representations_push(&mut selected_receipts,&mut selected_representations,representation,receipt);
    }
    // Rebuild parallel digest arrays in selected-ID order.
    let selected_representation_sha256:Vec<String>=selected_ids.iter().map(|id|{
        candidates[indices[id.as_str()]].representation_sha256.to_owned()
    }).collect();
    let selected_use_receipt_sha256:Vec<String>=selected_ids.iter().map(|id|{
        candidates[indices[id.as_str()]].receipt.canonical_sha256.clone()
    }).collect();
    excluded.sort_by(|a,b|a.evidence_id.cmp(&b.evidence_id));
    let mut selection=EvidenceSetSelectionReceiptV1{
        schema_version:EVIDENCE_SET_SCHEMA_V1.into(),context_sha256:ctx,candidate_set_sha256:cand,
        constraint_set_sha256:cons,constraint_policy_sha256:context.constraint_policy_sha256.clone(),
        selected_evidence_ids:selected_ids,selected_representation_sha256,selected_use_receipt_sha256,
        excluded,total_token_cost:cost,total_relevance_score_micros:score,max_token_cost:context.max_token_cost,
        canonical_sha256:String::new()
    };
    selection.canonical_sha256=selection.digest_without_commitment();
    Ok(selection)
}

fn selected_representations_push(
    _receipts:&mut Vec<String>,
    _representations:&mut Vec<String>,
    _representation:&str,
    _receipt:&str,
) {}
